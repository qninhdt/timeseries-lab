from typing import Any, Dict, Tuple, Type, List

import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule

from torchmetrics import MeanMetric, MaxMetric

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    PrecisionRecallCurve,
)

from layers import Normalize  # <<< Sử dụng Normalize đã sửa
from utils.augmentation import apply_augmentation

import wandb


class TraderLitModule(LightningModule):

    def __init__(
        self,
        model: Type[nn.Module],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        compile: bool = False,
        use_augmentation: bool = True,
        aug_jitter_prob: float = 0.0,
        aug_scaling_prob: float = 0.0,
        use_weighted_loss: bool = True,
    ) -> None:
        """
        Args:
            model (Type[nn.Module]): The neural network class (e.g., SimpleLSTM)
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            compile (bool): Enable torch.compile (PyTorch 2.0+ only)
            use_augmentation (bool): Enable data augmentation during training
            aug_jitter_prob (float): Probability of applying jitter augmentation
            aug_scaling_prob (float): Probability of applying scaling augmentation
            use_weighted_loss (bool): Enable coin-specific weighted loss for training
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model_class = model
        self.model = None

        # <<< MỚI: Đổi tên normalize_1h -> normalize (chỉ cần 1 module)
        self.normalize = None

        # Buffer để lưu trọng số loss, sẽ được khởi tạo trong setup()
        self.register_buffer("coin_pos_weights", None)

        # Store boolean mask for features normalized with close (norm_type=2)
        # Sẽ được tạo trong setup()
        self.close_norm_mask = None

        # --- Metrics for Training (Binary Classification) ---
        self.train_loss = MeanMetric()
        self.train_trade_acc = BinaryAccuracy()
        self.train_trade_ap = BinaryAveragePrecision()

        # --- Metrics for Val ---
        self.val_loss = MeanMetric()
        self.val_trade_acc = BinaryAccuracy()
        self.val_trade_ap = BinaryAveragePrecision()
        self.val_pr_curve = PrecisionRecallCurve(task="binary")

        self.val_epoch_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor,
        coin_ids: torch.Tensor = None,
        apply_augmentation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with multi-timeframe support.
        (Không thay đổi logic ở đây)
        """
        B, P, T_1h, D_1h = x_1h.shape
        _, _, T_4h, D_4h = x_4h.shape
        _, _, T_1d, D_1d = x_1d.shape

        # Reshape for normalization: (B, P, T, D) -> (B*P, T, D)
        x_1h_reshaped = x_1h.reshape(B * P, T_1h, D_1h)
        x_4h_reshaped = x_4h.reshape(B * P, T_4h, D_4h)
        x_1d_reshaped = x_1d.reshape(B * P, T_1d, D_1d)

        # Normalize all three timeframes together
        # <<< MỚI: self.normalize giờ đây xử lý cả 3
        x_1h_norm, x_4h_norm, x_1d_norm = self.normalize(
            x_1h_reshaped, x_4h_reshaped, x_1d_reshaped, mode="norm"
        )

        # Apply augmentation after normalization if in training mode
        if apply_augmentation and self.hparams.use_augmentation and self.training:
            # Chỉ augment 1h
            x_1h_norm = self._apply_augmentation(x_1h_norm)

        # Reshape back: (B*P, T, D) -> (B, P, T, D)
        x_1h_norm = x_1h_norm.reshape(B, P, T_1h, D_1h)
        x_4h_norm = x_4h_norm.reshape(B, P, T_4h, D_4h)
        x_1d_norm = x_1d_norm.reshape(B, P, T_1d, D_1d)

        return self.model(x_1h_norm, x_4h_norm, x_1d_norm, coin_ids=coin_ids, **kwargs)

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation only to features that are normalized with close (norm_type=2).
        x shape: (B*P, T, D_1h)

        (Không thay đổi logic ở đây)
        """
        x_aug = apply_augmentation(
            x,
            feature_mask=self.close_norm_mask,  # Mask này đã được tạo trong setup()
            jitter_prob=self.hparams.aug_jitter_prob,
            scaling_prob=self.hparams.aug_scaling_prob,
            jitter_sigma=0.1,
            scaling_sigma=0.1,
        )

        return x_aug

    def on_train_start(self) -> None:
        # (Không thay đổi)
        self.val_loss.reset()
        self.val_trade_acc.reset()
        self.val_trade_ap.reset()
        self.val_pr_curve.reset()
        self.val_epoch_outputs.clear()

    def model_step(
        self, batch: Dict[str, torch.Tensor], apply_augmentation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # (Không thay đổi)
        features_1h = batch["features_1h"]
        features_4h = batch["features_4h"]
        features_1d = batch["features_1d"]
        targets = batch["labels_trade"]
        coin_ids = batch["coin_ids"]

        logits = self.forward(
            features_1h,
            features_4h,
            features_1d,
            coin_ids,
            apply_augmentation=apply_augmentation,
        )
        logits = logits.squeeze(-1)

        targets_float = targets.float()

        if self.hparams.use_weighted_loss and self.coin_pos_weights is not None:
            batch_pos_weights = self.coin_pos_weights[coin_ids]
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_float, pos_weight=batch_pos_weights
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets_float)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        return loss, preds, probs, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # (Không thay đổi)
        loss, preds, probs, targets = self.model_step(batch, apply_augmentation=True)

        self.train_loss(loss)
        self.train_trade_acc(preds, targets)
        self.train_trade_ap(probs, targets)

        self.log(
            "train/trade_loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/trade_ap",
            self.train_trade_ap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/trade_acc",
            self.train_trade_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # (Không thay đổi)
        loss, preds, probs, targets = self.model_step(batch, apply_augmentation=False)

        self.val_loss(loss)
        self.val_trade_acc(preds, targets)
        self.val_trade_ap(probs, targets)
        self.val_pr_curve.update(probs, targets)

        self.val_epoch_outputs.append((probs.detach(), targets.detach()))

        self.log(
            "val/trade_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/trade_ap",
            self.val_trade_ap,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/trade_acc",
            self.val_trade_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        # (Không thay đổi)
        precisions, recalls, thresholds = self.val_pr_curve.compute()

        if not self.trainer.sanity_checking and self.logger is not None:
            loggers = self.logger if isinstance(self.logger, list) else [self.logger]
            wandb_logger = None
            for logger in loggers:
                if "WandbLogger" in logger.__class__.__name__:
                    wandb_logger = logger
                    break

            if wandb_logger is not None:
                log_payload = {}
                precisions_np = precisions.cpu().numpy()
                recalls_np = recalls.cpu().numpy()

                pr_table = wandb.Table(
                    data=[
                        [float(r), float(p)] for r, p in zip(recalls_np, precisions_np)
                    ],
                    columns=["Recall", "Precision"],
                )
                log_payload["val/pr_curve"] = wandb.plot.line(
                    pr_table, "Recall", "Precision", title="PR Curve"
                )

                if not self.val_epoch_outputs:
                    wandb_logger.experiment.log(log_payload)
                    self.val_pr_curve.reset()
                    return

                all_probs = torch.cat(
                    [item[0] for item in self.val_epoch_outputs], dim=0
                )
                all_targets = torch.cat(
                    [item[1] for item in self.val_epoch_outputs], dim=0
                )

                self.val_epoch_outputs.clear()

                all_probs_flat = all_probs.cpu().numpy().flatten()
                log_payload["val/prob_distribution"] = wandb.Histogram(all_probs_flat)

                datamodule = self.trainer.datamodule
                num_coins = all_probs.shape[1]
                coin_baselines = datamodule.coin_baselines
                val_coin_names = datamodule.val_coin_names

                per_coin_ap_metric = BinaryAveragePrecision().to(self.device)
                ap_table_data = []

                for i in range(num_coins):
                    coin_name = val_coin_names[i]
                    coin_probs = all_probs[:, i]
                    coin_targets = all_targets[:, i]

                    model_ap = per_coin_ap_metric(coin_probs, coin_targets.int()).item()
                    per_coin_ap_metric.reset()

                    baseline_ap = coin_baselines.get(coin_name, 0.0)

                    lift = model_ap / baseline_ap if baseline_ap > 0 else 0.0
                    lift_str = f"{lift:.2f}x"

                    ap_table_data.append([coin_name, model_ap, baseline_ap, lift_str])

                ap_table_data.sort(key=lambda x: x[1], reverse=True)

                ap_table = wandb.Table(
                    data=ap_table_data,
                    columns=["Coin", "AP", "Baseline AP", "Lift"],
                )
                log_payload["val/trade_ap_table"] = ap_table

                wandb_logger.experiment.log(log_payload)

        self.val_pr_curve.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        """
        <<< MỚI: Hàm setup đã được dọn dẹp.
        """
        datamodule = self.trainer.datamodule

        # Instantiate model (Không thay đổi)
        if self.model is None:
            max_coins = len(datamodule.coins)
            n_features_1h = datamodule.n_features["1h"]
            n_features_4h = datamodule.n_features["4h"]
            n_features_1d = datamodule.n_features["1d"]

            self.model = self.model_class(
                n_features_1h=n_features_1h,
                n_features_4h=n_features_4h,
                n_features_1d=n_features_1d,
                max_coins=max_coins,
            )

        # Calculate coin-specific positive weights for loss (Không thay đổi)
        if self.hparams.use_weighted_loss and self.coin_pos_weights is None:
            print("Calculating coin-specific positive weights for loss...")
            all_coin_names = datamodule.coins
            baselines = datamodule.coin_baselines
            n_coins = len(all_coin_names)
            pos_weights = torch.ones(n_coins, dtype=torch.float32)

            for i, coin_name in enumerate(all_coin_names):
                p = baselines[coin_name]
                epsilon = 1e-6
                p_stable = np.clip(p, epsilon, 1.0 - epsilon)
                pos_weights[i] = (1.0 - p_stable) / p_stable

            self.register_buffer("coin_pos_weights", pos_weights)
            print(
                f"Coin weights calculated and registered. Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}, Mean: {pos_weights.mean():.2f}"
            )

        # <<< MỚI: Khởi tạo Normalization module ---

        # 1. Lấy tất cả thông tin cần thiết từ datamodule
        n_features_1h = datamodule.n_features["1h"]
        n_features_4h = datamodule.n_features["4h"]
        n_features_1d = datamodule.n_features["1d"]

        norm_types_1h = datamodule.norm_types["1h"]
        norm_types_4h = datamodule.norm_types["4h"]
        norm_types_1d = datamodule.norm_types["1d"]

        close_idx_1h = datamodule.close_idx_1h

        # 2. Khởi tạo module Normalize với đầy đủ thông tin
        self.normalize = Normalize(
            num_features_1h=n_features_1h,
            num_features_4h=n_features_4h,
            num_features_1d=n_features_1d,
            norm_types_1h=norm_types_1h,
            norm_types_4h=norm_types_4h,
            norm_types_1d=norm_types_1d,
            close_idx_1h=close_idx_1h,
            affine=True,  # Giữ affine=True
        )

        # 3. Tạo boolean mask cho augmentation (chỉ dựa trên 1h)
        # Mask này dùng để augment các feature 1h có norm_type=2
        self.close_norm_mask = torch.tensor(
            [norm_type == 2 for norm_type in norm_types_1h],
            dtype=torch.bool,
            device=self.device,
        )
        n_close_norm = self.close_norm_mask.sum().item()
        print(
            f"Features (1h) normalized with close (will be augmented): {n_close_norm}/{n_features_1h}"
        )

        # --- Kết thúc khởi tạo Normalization ---

        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        # (Không thay đổi)
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0.000001,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
