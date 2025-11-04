from typing import Any, Dict, Tuple, Type, Callable, List

import torch
import torch.nn as nn
from lightning import LightningModule

from torchmetrics import MeanMetric, MaxMetric

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    PrecisionRecallCurve,
)

from crypto_datamodule import FEATURE_CONFIG, CryptoDataModule
from loss.binary_focal_loss import BinaryFocalLoss

import wandb


class TraderLitModule(LightningModule):

    def __init__(
        self,
        model: Type[nn.Module],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        compile: bool = False,
        use_focal_loss: bool = False,
        gamma: float = 2.0,
    ) -> None:
        """
        Args:
            model (Type[nn.Module]): The neural network class (e.g., SimpleLSTM)
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            compile (bool): Enable torch.compile (PyTorch 2.0+ only)
            use_focal_loss (bool): If True, use focal loss with alpha from class weights
                                  If False, use BCE with class weights (default)
            gamma (float): Focal loss gamma parameter (default: 2.0)
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])

        n_features = len(FEATURE_CONFIG)
        # Model will be instantiated in setup() after we have access to datamodule
        self.model_class = model
        self.model = None

        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

        # Loss function will be initialized in on_train_start after getting class weights
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None

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

    def forward(self, x: torch.Tensor, coin_ids: torch.Tensor = None) -> torch.Tensor:
        # Try to pass coin_ids if the model supports it
        try:
            return self.model(x, coin_ids)
        except TypeError:
            # If model doesn't accept coin_ids, fall back to just x
            return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_trade_acc.reset()
        self.val_trade_ap.reset()
        self.val_pr_curve.reset()
        self.val_epoch_outputs.clear()

        # Logic loss sẽ được xử lý trong model_step sử dụng per-coin weights
        # Chỉ khởi tạo loss_fn nếu dùng Focal Loss
        if self.use_focal_loss:
            # Focal loss với alpha toàn cục (rút từ class_weights nếu có)
            global_alpha = 0.5
            if self.trainer is not None:
                datamodule = self.trainer.datamodule
                if (
                    isinstance(datamodule, CryptoDataModule)
                    and getattr(datamodule, "class_weights", None) is not None
                ):
                    class_weights = datamodule.class_weights.to(self.device)
                    total_weight = class_weights[0] + class_weights[1]
                    global_alpha = (class_weights[1] / total_weight).item()

            global_alpha = 0.5
            focal_loss = BinaryFocalLoss(
                alpha=global_alpha, gamma=self.gamma, reduction="mean"
            )
            self.loss_fn = focal_loss
            print(
                f"Using Focal Loss (Global Alpha): alpha={global_alpha:.4f}, gamma={self.gamma:.4f}"
            )
        else:
            # Sẽ sử dụng per-coin weight BCE trong model_step
            self.loss_fn = None
            print("Using Per-Coin Weighted BCE (weights from batch)")

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["features"]  # (B, P, T, D)
        targets = batch["labels_trade"]  # (B, P), bool type
        coin_ids = batch.get("coin_ids", None)  # (B, P), coin indices

        # Lấy pos_weights từ batch
        pos_weights = batch.get("pos_weights", None)  # (B, P)

        logits = self.forward(features, coin_ids)  # (B, P, 1)
        logits = logits.squeeze(-1)  # (B, P)

        targets_float = targets.float()

        # Tính toán loss
        if self.use_focal_loss and self.loss_fn is not None:
            # Focal Loss (alpha toàn cục)
            loss = self.loss_fn(logits, targets_float)
        else:
            # Per-coin weighted BCE
            if pos_weights is None:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, targets_float
                )
            else:
                pos_weights_device = pos_weights.to(logits.device)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    targets_float,  # pos_weight=pos_weights_device
                )

        # Get probabilities and binary predictions for metrics
        probs = torch.sigmoid(logits)  # (B, P)
        preds = (probs > 0.5).float()  # (B, P)

        return loss, preds, probs, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, probs, targets = self.model_step(batch)

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
        loss, preds, probs, targets = self.model_step(batch)

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

                # Lấy dictionary baseline từ datamodule
                coin_baselines = getattr(datamodule, "coin_baselines", {})

                num_coins = all_probs.shape[1]
                val_coin_names = []

                if (
                    isinstance(datamodule, CryptoDataModule)
                    and datamodule.val_coin_names is not None
                    and len(datamodule.val_coin_names) == num_coins
                ):
                    val_coin_names = datamodule.val_coin_names
                else:
                    val_coin_names = [f"coin_{i}" for i in range(num_coins)]

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
        # Instantiate model with correct max_coins from datamodule
        if self.model is None:
            n_features = len(FEATURE_CONFIG)
            max_coins = (
                len(self.trainer.datamodule.coins)
                if hasattr(self.trainer.datamodule, "coins")
                else 128
            )
            self.model = self.model_class(n_features=n_features, max_coins=max_coins)

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
