from typing import Any, Dict, Tuple, Type, List

import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule

from torchmetrics import MeanMetric

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
    ) -> None:
        """
        Args:
            model (Type[nn.Module]): The neural network class (e.g., LSTM)
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for optimizer
            compile (bool): Enable torch.compile (PyTorch 2.0+ only)
            use_augmentation (bool): Enable data augmentation during training
            aug_jitter_prob (float): Probability of applying jitter augmentation
            aug_scaling_prob (float): Probability of applying scaling augmentation
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model_class = model
        self.model = None
        self.normalize = None
        self.close_norm_mask = None

        # --- Metrics for Training ---
        self.train_loss = MeanMetric()

        # --- Metrics for Val ---
        self.val_loss = MeanMetric()

        # Store outputs for histogram (beta only)
        self.val_epoch_outputs: List[torch.Tensor] = []

    @staticmethod
    def _profit_loss(
        beta: torch.Tensor,
        close_t: torch.Tensor,
        close_t_plus_1: torch.Tensor,
        current_epoch: int,
    ) -> torch.Tensor:
        """
        Profit-based loss using beta directly.
        loss = -(return * beta)

        Args:
            beta: (B, 1) trading signal in [-1, 1]
            close_t: Close prices at time t (B,)
            close_t_plus_1: Close prices at time t+1 (B,)
            current_epoch: Current training epoch

        Returns:
            loss: Negative profit (mean over batch)
            profit: Profit per sample
        """
        # Calculate price change (not percentage)
        returns = (close_t_plus_1 - close_t) / close_t  # (B,)

        # Squeeze beta to (B,)
        beta_squeezed = beta.squeeze(-1)  # (B,)

        # Profit = return * beta - regularization
        profit = returns * beta_squeezed - 0.001 * beta_squeezed.abs()

        # Loss = negative profit
        loss = -profit.mean()  # - 0.001 * beta.std()

        return loss, profit

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
        Forward pass through normalization and model.

        Returns:
            beta: (B, 1) trading signal in [-1, 1]
        """
        B, T_1h, D_1h = x_1h.shape
        _, T_4h, D_4h = x_4h.shape
        _, T_1d, D_1d = x_1d.shape

        x_1h_reshaped = x_1h
        x_4h_reshaped = x_4h
        x_1d_reshaped = x_1d

        x_1h_norm, x_4h_norm, x_1d_norm = self.normalize(
            x_1h_reshaped, x_4h_reshaped, x_1d_reshaped, mode="norm"
        )

        if apply_augmentation and self.hparams.use_augmentation and self.training:
            x_1h_norm = self._apply_augmentation(x_1h_norm)

        # Model returns beta (B, 1)
        beta = self.model(x_1h_norm, x_4h_norm, x_1d_norm, coin_ids=coin_ids, **kwargs)
        return beta

    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        (Không thay đổi logic ở đây)
        """
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
        self.val_epoch_outputs.clear()

    def model_step(
        self, batch: Dict[str, torch.Tensor], apply_augmentation: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: Batch containing features and close prices
            apply_augmentation: Whether to apply augmentation

        Returns:
            loss: Profit-based loss
            beta: (B, 1) trading signal in [-1, 1]
            profit: (B,) profit per sample
        """
        features_1h = batch["features_1h"]
        features_4h = batch["features_4h"]
        features_1d = batch["features_1d"]
        close_t = batch["close_t"].to(self.device)
        close_t_plus_1 = batch["close_t_plus_1"].to(self.device)
        coin_ids = batch["coin_ids"]

        beta = self.forward(
            features_1h,
            features_4h,
            features_1d,
            coin_ids,
            apply_augmentation=apply_augmentation,
        )

        # Profit-based loss
        loss, profit = self._profit_loss(
            beta, close_t, close_t_plus_1, self.current_epoch
        )

        return loss, beta, profit

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, beta, profit = self.model_step(batch, apply_augmentation=True)

        self.train_loss(loss)

        # Calculate profit statistics
        profit_mean = profit.mean()

        # Calculate beta statistics
        beta_mean = beta.mean()
        beta_std = beta.std()

        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        self.log(
            "train/profit",
            profit_mean,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        self.log(
            "train/beta_mean",
            beta_mean,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        self.log(
            "train/beta_std",
            beta_std,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, beta, profit = self.model_step(batch, apply_augmentation=False)

        self.val_loss(loss)

        # Calculate profit statistics
        profit_mean = profit.mean()

        # Store outputs for histogram (beta only)
        self.val_epoch_outputs.append(beta.detach())

        # Calculate beta statistics
        beta_mean = beta.mean()
        beta_std = beta.std()

        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "val/profit",
            profit_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "val/beta_mean",
            beta_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "val/beta_std",
            beta_std,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking and self.logger is not None:
            loggers = self.logger if isinstance(self.logger, list) else [self.logger]
            wandb_logger = None
            for logger in loggers:
                if "WandbLogger" in logger.__class__.__name__:
                    wandb_logger = logger
                    break

            if wandb_logger is not None and self.val_epoch_outputs:
                # Concatenate all beta
                all_betas = torch.cat(self.val_epoch_outputs, dim=0)  # (N,)

                # Create histogram
                log_payload = {
                    "val/beta_distribution": wandb.Histogram(
                        all_betas.cpu().numpy().flatten()
                    ),
                }

                wandb_logger.experiment.log(log_payload)

        # Clear outputs
        self.val_epoch_outputs.clear()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        """
        Setup model and normalization layers.
        """
        datamodule = self.trainer.datamodule

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

        # Khởi tạo Normalization module
        if self.normalize is None:
            n_features_1h = datamodule.n_features["1h"]
            n_features_4h = datamodule.n_features["4h"]
            n_features_1d = datamodule.n_features["1d"]
            norm_types_1h = datamodule.norm_types["1h"]
            norm_types_4h = datamodule.norm_types["4h"]
            norm_types_1d = datamodule.norm_types["1d"]
            close_idx_1h = datamodule.close_idx_1h

            self.normalize = Normalize(
                num_features_1h=n_features_1h,
                num_features_4h=n_features_4h,
                num_features_1d=n_features_1d,
                norm_types_1h=norm_types_1h,
                norm_types_4h=norm_types_4h,
                norm_types_1d=norm_types_1d,
                close_idx_1h=close_idx_1h,
                affine=True,
            )

            self.close_norm_mask = torch.tensor(
                [norm_type == 2 for norm_type in norm_types_1h],
                dtype=torch.bool,
                device=self.device,
            )
            n_close_norm = self.close_norm_mask.sum().item()
            print(
                f"Features (1h) normalized with close (will be augmented): {n_close_norm}/{n_features_1h}"
            )

        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Linear scheduler with warmup
        warmup_epochs = 5
        total_epochs = self.trainer.max_epochs

        # Warmup: 0.01 -> 1.0 over warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # Linear decay: 1.0 -> 0 over remaining epochs
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=total_epochs - warmup_epochs,
        )

        # Combine warmup and decay
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
