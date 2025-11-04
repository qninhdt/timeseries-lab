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

from crypto_datamodule_v2 import FEATURE_CONFIG
from layers import Normalize
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
        # save_hyperparameters sẽ tự động lấy use_weighted_loss
        self.save_hyperparameters(logger=False, ignore=["model"])

        n_features = len(FEATURE_CONFIG)
        # Model will be instantiated in setup() after we have access to datamodule
        self.model_class = model
        self.model = None

        # Normalization module will be initialized in setup()
        self.normalize = None

        # Buffer để lưu trọng số loss, sẽ được khởi tạo trong setup()
        self.register_buffer("coin_pos_weights", None)

        # Store boolean mask for features normalized with close (norm_type=2)
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
        x: torch.Tensor,
        coin_ids: torch.Tensor = None,
        apply_augmentation: bool = False,
    ) -> torch.Tensor:
        # Normalize features: (B, P, T, D) -> (B*P, T, D) -> normalize -> (B, P, T, D)
        B, P, T, D = x.shape
        x_reshaped = x.reshape(B * P, T, D)
        x_normalized = self.normalize(x_reshaped, mode="norm")

        # Apply augmentation after normalization if in training mode
        if apply_augmentation and self.hparams.use_augmentation and self.training:
            x_normalized = self._apply_augmentation(x_normalized)

        x_normalized = x_normalized.reshape(B, P, T, D)

        # Try to pass coin_ids if the model supports it
        try:
            return self.model(x_normalized, coin_ids)
        except TypeError:
            # If model doesn't accept coin_ids, fall back to just x
            return self.model(x_normalized)

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation only to features that are normalized with close (norm_type=2).
        x shape: (B*P, T, D)
        Uses PyTorch operations directly on GPU without CPU conversion.

        Only applies 2 augmentations:
        - jitter: add Gaussian noise
        - scaling: multiply by random factors
        """
        if self.close_norm_mask is None:
            return x

        # Apply augmentation with feature mask
        # feature_mask is boolean tensor indicating which features to augment
        x_aug = apply_augmentation(
            x,
            feature_mask=self.close_norm_mask,
            jitter_prob=self.hparams.aug_jitter_prob,
            scaling_prob=self.hparams.aug_scaling_prob,
            jitter_sigma=0.1,
            scaling_sigma=0.1,
        )

        return x_aug

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_trade_acc.reset()
        self.val_trade_ap.reset()
        self.val_pr_curve.reset()
        self.val_epoch_outputs.clear()

    def model_step(
        self, batch: Dict[str, torch.Tensor], apply_augmentation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["features"]  # (B, P, T, D)
        targets = batch["labels_trade"]  # (B, P), bool type
        coin_ids = batch.get("coin_ids", None)  # (B, P), coin indices

        logits = self.forward(
            features, coin_ids, apply_augmentation=apply_augmentation
        )  # (B, P, 1)
        logits = logits.squeeze(-1)  # (B, P)

        targets_float = targets.float()

        # --- THAY ĐỔI TÍNH TOÁN LOSS ---
        if self.hparams.use_weighted_loss and self.training and coin_ids is not None:
            if self.coin_pos_weights is None:
                # Fallback nếu setup() chưa chạy (không nên xảy ra)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_float
                )
            else:
                # self.coin_pos_weights có shape (C_total)
                # coin_ids có shape (B, P)
                # Lấy ra các trọng số tương ứng cho batch này
                batch_pos_weights = self.coin_pos_weights[coin_ids]  # Shape (B, P)

                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_float, pos_weight=batch_pos_weights
                )
        else:
            # Logic cũ: Loss BCE không trọng số (dùng cho validation hoặc khi tắt weighted_loss)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets_float)
        # --- KẾT THÚC THAY ĐỔI ---

        # Get probabilities and binary predictions for metrics
        probs = torch.sigmoid(logits)  # (B, P)
        preds = (probs > 0.5).float()  # (B, P)

        return loss, preds, probs, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, probs, targets = self.model_step(batch, apply_augmentation=True)

        # Update and log metrics
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
        # model_step sẽ tự động dùng loss không trọng số vì self.training là False
        loss, preds, probs, targets = self.model_step(batch, apply_augmentation=False)

        # Update metrics
        self.val_loss(loss)
        self.val_trade_acc(preds, targets)
        self.val_trade_ap(probs, targets)
        self.val_pr_curve.update(probs, targets)

        # Store probs for histogram logging
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
        # 1. Tính toán PR curve tổng hợp (metric cũ)
        precisions, recalls, thresholds = self.val_pr_curve.compute()

        # 2. Kiểm tra nếu không phải sanity check và có logger
        if not self.trainer.sanity_checking and self.logger is not None:
            loggers = self.logger if isinstance(self.logger, list) else [self.logger]
            wandb_logger = None
            for logger in loggers:
                if (
                    hasattr(logger, "__class__")
                    and "WandbLogger" in logger.__class__.__name__
                ):
                    wandb_logger = logger
                    break

            if wandb_logger is not None and hasattr(wandb_logger, "experiment"):

                # --- A. Xử lý PR Curve (Logic cũ) ---
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

                # --- B. Xử lý Histogram & Bảng AP ---
                if not self.val_epoch_outputs:
                    # Nếu không có output, chỉ log PR curve
                    wandb_logger.experiment.log(log_payload)
                    self.val_pr_curve.reset()
                    return

                # Tổng hợp tất cả outputs từ các validation step
                all_probs = torch.cat(
                    [item[0] for item in self.val_epoch_outputs], dim=0
                )  # (N_samples, P_coins)
                all_targets = torch.cat(
                    [item[1] for item in self.val_epoch_outputs], dim=0
                )  # (N_samples, P_coins)

                # Xóa danh sách cho epoch tiếp theo
                self.val_epoch_outputs.clear()

                # Log histogram
                all_probs_flat = all_probs.cpu().numpy().flatten()
                log_payload["val/prob_distribution"] = wandb.Histogram(all_probs_flat)

                # --- LOGIC MỚI: TẠO BẢNG AP (MODEL vs BASELINE) ---
                datamodule = self.trainer.datamodule

                num_coins = all_probs.shape[1]
                coin_baselines = datamodule.coin_baselines  # Dict[str, float]
                val_coin_names = datamodule.val_coin_names  # List of coin names

                per_coin_ap_metric = BinaryAveragePrecision().to(self.device)
                ap_table_data = []

                for i in range(num_coins):
                    coin_name = val_coin_names[i]
                    coin_probs = all_probs[:, i]
                    coin_targets = all_targets[:, i]

                    # 1. Tính Model AP
                    model_ap = per_coin_ap_metric(coin_probs, coin_targets.int()).item()
                    per_coin_ap_metric.reset()

                    # 2. Lấy Baseline AP (từ datamodule)
                    baseline_ap = coin_baselines.get(coin_name, 0.0)

                    # 3. Tính Lift
                    lift = model_ap / baseline_ap if baseline_ap > 0 else 0.0
                    lift_str = f"{lift:.2f}x"

                    ap_table_data.append([coin_name, model_ap, baseline_ap, lift_str])

                # Sắp xếp bảng theo Model AP giảm dần
                ap_table_data.sort(key=lambda x: x[1], reverse=True)

                # Tạo bảng wandb với các cột mới
                ap_table = wandb.Table(
                    data=ap_table_data,
                    columns=["Coin", "AP", "Baseline AP", "Lift"],
                )
                log_payload["val/trade_ap_table"] = ap_table
                # --- KẾT THÚC LOGIC MỚI ---

                # Log tất cả lên wandb
                wandb_logger.experiment.log(log_payload)

        # Reset metric tổng hợp cho epoch tiếp theo
        self.val_pr_curve.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        # Lấy datamodule
        datamodule = self.trainer.datamodule

        # Instantiate model with correct max_coins from datamodule
        if self.model is None:
            n_features = len(FEATURE_CONFIG)
            max_coins = len(datamodule.coins) if hasattr(datamodule, "coins") else 128
            self.model = self.model_class(n_features=n_features, max_coins=max_coins)

        # --- THÊM LOGIC TÍNH TRỌNG SỐ ---
        if self.hparams.use_weighted_loss and self.coin_pos_weights is None:
            print("Calculating coin-specific positive weights for loss...")
            if not hasattr(datamodule, "coins") or not hasattr(
                datamodule, "coin_baselines"
            ):
                raise AttributeError(
                    "DataModule must have 'coins' list and 'coin_baselines' dict to use weighted loss."
                )

            all_coin_names = datamodule.coins
            baselines = datamodule.coin_baselines
            n_coins = len(all_coin_names)

            # Khởi tạo tensor trọng số
            pos_weights = torch.ones(n_coins, dtype=torch.float32)

            for i, coin_name in enumerate(all_coin_names):
                # Lấy tỉ lệ positive (baseline)
                p = baselines.get(coin_name, 0.5)  # Mặc định 0.5 (trọng số 1) nếu thiếu

                # Tính trọng số = (1-p) / p
                # Thêm epsilon để đảm bảo ổn định số học
                epsilon = 1e-6
                p_stable = np.clip(p, epsilon, 1.0 - epsilon)
                pos_weights[i] = (1.0 - p_stable) / p_stable

            # Đăng ký làm buffer để tự động chuyển sang device (GPU/CPU)
            self.register_buffer("coin_pos_weights", pos_weights)
            print(
                f"Coin weights calculated and registered. Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}, Mean: {pos_weights.mean():.2f}"
            )
        # --- KẾT THÚC LOGIC TRỌNG SỐ ---

        # Initialize normalization module
        if self.normalize is None:
            # Get feature names and norm types from config
            feature_names = list(FEATURE_CONFIG.keys())
            norm_types = [
                FEATURE_CONFIG[name].get("norm_type", 0) for name in feature_names
            ]
            close_idx = feature_names.index("close")

            n_features = len(FEATURE_CONFIG)
            self.normalize = Normalize(
                num_features=n_features,
                norm_types=norm_types,
                close_idx=close_idx,
                affine=True,
            )

            # Create boolean mask for features normalized with close (norm_type=2)
            # This will be used for selective augmentation
            self.close_norm_mask = torch.tensor(
                [norm_type == 2 for norm_type in norm_types],
                dtype=torch.bool,
                device=self.device,
            )
            n_close_norm = self.close_norm_mask.sum().item()
            print(
                f"Features normalized with close (will be augmented): {n_close_norm}/{n_features}"
            )

        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
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
