from typing import Any, Dict, Tuple, Type

import torch
import torch.nn as nn
from lightning import LightningModule

from torchmetrics import MeanMetric, MaxMetric

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from crypto_datamodule import FEATURE_CONFIG


class TraderLitModule(LightningModule):

    def __init__(
        self,
        model: Type[nn.Module],
        trade_loss: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        compile: bool = False,
    ) -> None:
        """
        Args:
            model_class (Type[nn.Module]): The neural network class (e.g., SimpleLSTM)
            trade_loss (nn.Module): The loss function (e.g., BinaryFocalLoss)
            optimizer (torch.optim.Optimizer): The optimizer
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler
            compile (bool): Enable torch.compile (PyTorch 2.0+ only)
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model", "trade_loss"])

        n_features = len(FEATURE_CONFIG)
        self.model = model(n_features=n_features)

        self.trade_loss = trade_loss

        # --- Metrics for Training (Binary Classification) ---
        self.train_loss = MeanMetric()
        self.train_trade_acc = BinaryAccuracy()
        self.train_trade_f1 = BinaryF1Score()
        self.train_trade_precision = BinaryPrecision()
        self.train_trade_recall = BinaryRecall()

        # --- Metrics for Validation ---
        self.val_loss = MeanMetric()
        self.val_trade_acc = BinaryAccuracy()
        self.val_trade_f1 = BinaryF1Score()
        self.val_trade_precision = BinaryPrecision()
        self.val_trade_recall = BinaryRecall()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_trade_acc.reset()
        self.val_trade_f1.reset()
        self.val_trade_precision.reset()
        self.val_trade_recall.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = batch["features"]  # (B, P, T, D)
        targets = batch["labels_trade"]  # (B, P), bool type

        logits = self.forward(features)  # (B, P, 1)
        logits = logits.squeeze(-1)  # (B, P)

        # Calculate loss (requires targets as float)
        loss = self.trade_loss(logits, targets.float())

        # Get probabilities for metrics
        preds = torch.sigmoid(logits)

        return loss, preds, targets

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # Update and log metrics
        self.train_loss(loss)
        self.train_trade_acc(preds, targets)
        self.train_trade_f1(preds, targets)
        self.train_trade_precision(preds, targets)
        self.train_trade_recall(preds, targets)

        self.log(
            "train/trade_loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "train/trade_f1",
            self.train_trade_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/trade_precision",
            self.train_trade_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/trade_recall",
            self.train_trade_recall,
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
        loss, preds, targets = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        self.val_trade_acc(preds, targets)
        self.val_trade_f1(preds, targets)
        self.val_trade_precision(preds, targets)
        self.val_trade_recall(preds, targets)

        self.log(
            "val/trade_loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/trade_f1",
            self.val_trade_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/trade_precision",
            self.val_trade_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/trade_recall",
            self.val_trade_recall,
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
        pass

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
