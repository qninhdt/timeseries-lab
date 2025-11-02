from typing import Any, Dict, Tuple, Type, Callable

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
        self.model = model(n_features=n_features)

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

        # Store probs for histogram logging
        self.val_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_trade_acc.reset()
        self.val_trade_ap.reset()
        self.val_pr_curve.reset()
        self.val_step_outputs.clear()

        # Initialize loss function based on use_focal_loss flag
        if self.trainer is not None:
            datamodule = self.trainer.datamodule
            if (
                isinstance(datamodule, CryptoDataModule)
                and datamodule.class_weights is not None
            ):
                class_weights = datamodule.class_weights.to(self.device)

                if self.use_focal_loss:
                    # Calculate alpha for focal loss from class weights
                    # alpha = weight_class_1 / (weight_class_0 + weight_class_1)
                    total_weight = class_weights[0] + class_weights[1]
                    focal_alpha = (class_weights[1] / total_weight).item()

                    focal_loss = BinaryFocalLoss(
                        alpha=focal_alpha, gamma=self.gamma, reduction="mean"
                    )
                    self.loss_fn = focal_loss

                    print(
                        f"Using Focal Loss: alpha={focal_alpha:.4f}, gamma={self.gamma:.4f}"
                    )
                    print(f"Class weights: {class_weights}")
                else:
                    # Use BCE with class weights
                    # pos_weight = weight_class_1 / weight_class_0
                    pos_weight = (class_weights[1] / class_weights[0]).to(self.device)

                    # Create closure with pos_weight captured
                    def make_bce_loss_fn(pos_w: torch.Tensor):
                        def bce_loss_fn(
                            logits: torch.Tensor, targets: torch.Tensor
                        ) -> torch.Tensor:
                            return nn.functional.binary_cross_entropy_with_logits(
                                logits, targets.float(), pos_weight=pos_w
                            )

                        return bce_loss_fn

                    self.loss_fn = make_bce_loss_fn(pos_weight)

                    print(f"Using BCE with pos_weight={pos_weight.item():.4f}")
                    print(f"Class weights: {class_weights}")
            else:
                # Fallback: standard loss without weights
                if self.use_focal_loss:
                    focal_loss = BinaryFocalLoss(
                        alpha=0.5, gamma=self.gamma, reduction="mean"
                    )
                    self.loss_fn = focal_loss
                    print(
                        f"Using Focal Loss (no class weights): alpha=0.5, gamma={self.gamma:.4f}"
                    )
                else:
                    # Standard BCE without weights
                    self.loss_fn = lambda logits, targets: nn.functional.binary_cross_entropy_with_logits(
                        logits, targets.float()
                    )
                    print("Using standard BCE (no class weights available)")

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["features"]  # (B, P, T, D)
        targets = batch["labels_trade"]  # (B, P), bool type

        logits = self.forward(features)  # (B, P, 1)
        logits = logits.squeeze(-1)  # (B, P)

        # Calculate loss using initialized loss function
        if self.loss_fn is None:
            # Fallback if loss_fn not initialized
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets.float()
            )
        else:
            loss = self.loss_fn(logits, targets.float())

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
        self.val_step_outputs.append(probs.detach())

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
        precisions, recalls, thresholds = self.val_pr_curve.compute()

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
                precisions = precisions.cpu().numpy()
                recalls = recalls.cpu().numpy()

                # Create data table for wandb
                table = wandb.Table(
                    data=[[float(r), float(p)] for r, p in zip(recalls, precisions)],
                    columns=["Recall", "Precision"],
                )

                # Log probability distribution histogram
                all_probs = torch.cat(self.val_step_outputs).cpu()

                wandb_logger.experiment.log(
                    {
                        "val/pr_curve": wandb.plot.line(
                            table,
                            "Recall",
                            "Precision",
                            title="PR Curve",
                        ),
                        "val/prob_distribution": wandb.Histogram(all_probs.numpy()),
                    },
                )

                # Clear the list for next epoch
                self.val_step_outputs.clear()

        # Reset metric for next epoch
        self.val_pr_curve.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
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
