from typing import Any, Dict, Tuple, Type, List

import torch
import torch.nn as nn
import numpy as np
from lightning import LightningModule

from torchmetrics import MeanMetric, MaxMetric

# <<< THAY ĐỔI: Import metrics đa lớp
from torchmetrics.classification import (
    MulticlassAveragePrecision,
    MulticlassPrecisionRecallCurve,  # <<< THÊM VÀO
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
        use_weighted_loss: bool = False,  # <<< Lưu ý: Hparam này hiện không được sử dụng
        use_focal_loss: bool = True,
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
                                     (HIỆN KHÔNG CÓ TÁC DỤNG TRONG 3-CLASS)
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["model"])

        self.model_class = model
        self.model = None
        self.normalize = None
        self.register_buffer("coin_pos_weights", None)
        self.close_norm_mask = None

        # --- Metrics for Training ---
        self.train_loss = MeanMetric()
        self.train_ap = MulticlassAveragePrecision(num_classes=3, average=None)

        # --- Metrics for Val ---
        self.val_loss = MeanMetric()
        self.val_ap = MulticlassAveragePrecision(num_classes=3, average=None)
        # <<< THÊM VÀO: Metrics cho PR curve đa lớp
        self.val_pr_curve = MulticlassPrecisionRecallCurve(
            num_classes=3, thresholds=100
        )

        # <<< THAY ĐỔI: Cập nhật type hint (probs, labels, coin_ids, logits)
        self.val_epoch_outputs: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    @staticmethod
    def _focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Multiclass focal loss on logits with integer targets.
        """
        probs = torch.softmax(logits, dim=1)
        # gather p_t
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        log_pt = pt.log()
        loss = -((1 - pt) ** gamma) * log_pt
        if alpha is not None:
            # alpha per-class weights (shape [C])
            at = alpha.to(logits.device)[targets]
            loss = at * loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        return loss

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
        (Không thay đổi logic ở đây)
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

        # Giả định model trả về (B, 3)
        return self.model(x_1h_norm, x_4h_norm, x_1d_norm, coin_ids=coin_ids, **kwargs)

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
        # (Không thay đổi)
        self.val_loss.reset()
        self.val_ap.reset()
        self.val_pr_curve.reset()  # <<< THÊM VÀO
        self.val_epoch_outputs.clear()

    def model_step(
        self, batch: Dict[str, torch.Tensor], apply_augmentation: bool = False
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:  # <<< THAY ĐỔI

        features_1h = batch["features_1h"]
        features_4h = batch["features_4h"]
        features_1d = batch["features_1d"]
        # Targets: multiclass labels 0=hold,1=buy,2=sell
        labels_multi = batch["labels"].to(self.device)
        coin_ids = batch["coin_ids"]

        outputs = self.forward(
            features_1h,
            features_4h,
            features_1d,
            coin_ids,
            apply_augmentation=apply_augmentation,
        )
        logits = outputs  # (B,3)

        # single loss: CE or Focal over 3 classes
        if self.hparams.use_focal_loss:
            total_loss = self._focal_loss(logits, labels_multi)
        else:
            total_loss = nn.functional.cross_entropy(logits, labels_multi)

        # probs and preds
        probs = torch.softmax(logits, dim=1)  # (B,3)
        preds_multi = torch.argmax(probs, dim=1)  # (B,) 0=hold, 1=buy, 2=sell

        # Return everything needed downstream
        return (
            total_loss,
            preds_multi,
            labels_multi,
            probs,
            logits,  # <<< THÊM VÀO
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        (
            loss,
            preds_multi,
            labels_multi,
            probs,
            _logits,  # <<< THAY ĐỔI
        ) = self.model_step(batch, apply_augmentation=True)

        self.train_loss(loss)
        self.train_ap(probs, labels_multi)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
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

    def on_train_epoch_end(self) -> None:
        """
        Tính toán và log AP (buy, sell, mean) cho training.
        """
        all_aps = self.train_ap.compute()  # [AP_hold, AP_buy, AP_sell]
        if all_aps is not None and all_aps.numel() == 3:
            buy_ap = all_aps[1].item()
            sell_ap = all_aps[2].item()
            mean_ap = (buy_ap + sell_ap) / 2.0

            self.log(
                "train/buy_ap", buy_ap, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "train/sell_ap", sell_ap, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log("train/ap", mean_ap, on_step=False, on_epoch=True, prog_bar=True)

        # Reset metrics sau mỗi epoch
        self.train_ap.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:

        (
            loss,
            preds_multi,
            labels_multi,
            probs,
            logits,  # <<< THAY ĐỔI
        ) = self.model_step(batch, apply_augmentation=False)

        self.val_loss(loss)
        self.val_ap(probs, labels_multi)
        self.val_pr_curve(probs, labels_multi)  # <<< THÊM VÀO

        # <<< THAY ĐỔI: Thêm logits vào outputs
        self.val_epoch_outputs.append(
            (
                probs.detach(),
                labels_multi.detach(),
                batch["coin_ids"].detach(),
                logits.detach(),
            )
        )

        # unified val loss
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:

        # AP (buy, sell, mean) tổng thể
        all_val_aps = self.val_ap.compute()  # [AP_hold, AP_buy, AP_sell]
        if all_val_aps is not None and all_val_aps.numel() == 3:
            buy_ap = all_val_aps[1].item()
            sell_ap = all_val_aps[2].item()
            mean_ap = (buy_ap + sell_ap) / 2.0

            self.log("val/buy_ap", buy_ap, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val/sell_ap", sell_ap, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log("val/ap", mean_ap, on_step=False, on_epoch=True, prog_bar=True)

        # <<< THAY ĐỔI: Tính PR Curve (multiclass)
        pr_all_classes = self.val_pr_curve.compute()

        if not self.trainer.sanity_checking and self.logger is not None:
            loggers = self.logger if isinstance(self.logger, list) else [self.logger]
            wandb_logger = None
            for logger in loggers:
                if "WandbLogger" in logger.__class__.__name__:
                    wandb_logger = logger
                    break

            if wandb_logger is not None:
                log_payload = {}

                # <<< THÊM VÀO: Plot PR Curve cho Buy và Sell
                if pr_all_classes is not None:
                    # pr_all_classes[0] = precision (list of tensors)
                    # pr_all_classes[1] = recall (list of tensors)

                    # Tách riêng cho Buy (class 1) và Sell (class 2)
                    p_buy, r_buy = pr_all_classes[0][1], pr_all_classes[1][1]
                    p_sell, r_sell = pr_all_classes[0][2], pr_all_classes[1][2]

                    # Plot PR Curve cho Buy
                    pr_data_buy = []
                    if p_buy is not None and p_buy.numel() > 0:
                        p_vals = p_buy.cpu().numpy()
                        r_vals = r_buy.cpu().numpy()
                        pr_data_buy.extend(
                            [[float(r), float(p)] for r, p in zip(r_vals, p_vals)]
                        )

                    if pr_data_buy:
                        pr_table_buy = wandb.Table(
                            data=pr_data_buy, columns=["Recall", "Precision"]
                        )
                        log_payload["val/buy_pr_curve"] = wandb.plot.line(
                            pr_table_buy,
                            "Recall",
                            "Precision",
                            title="PR Curve (Buy)",
                        )

                    # Plot PR Curve cho Sell
                    pr_data_sell = []
                    if p_sell is not None and p_sell.numel() > 0:
                        p_vals = p_sell.cpu().numpy()
                        r_vals = r_sell.cpu().numpy()
                        pr_data_sell.extend(
                            [[float(r), float(p)] for r, p in zip(r_vals, p_vals)]
                        )

                    if pr_data_sell:
                        pr_table_sell = wandb.Table(
                            data=pr_data_sell, columns=["Recall", "Precision"]
                        )
                        log_payload["val/sell_pr_curve"] = wandb.plot.line(
                            pr_table_sell,
                            "Recall",
                            "Precision",
                            title="PR Curve (Sell)",
                        )
                # <<< KẾT THÚC THÊM VÀO

                if not self.val_epoch_outputs:
                    wandb_logger.experiment.log(log_payload)
                    self.val_ap.reset()
                    self.val_pr_curve.reset()  # <<< THÊM VÀO
                    return

                all_probs = torch.cat(
                    [item[0] for item in self.val_epoch_outputs], dim=0
                )
                all_labels_multi = torch.cat(
                    [item[1] for item in self.val_epoch_outputs], dim=0
                )
                all_coin_ids = torch.cat(
                    [item[2] for item in self.val_epoch_outputs], dim=0
                )
                # <<< THÊM VÀO: Thu thập logits
                all_logits = torch.cat(
                    [item[3] for item in self.val_epoch_outputs], dim=0
                )

                self.val_epoch_outputs.clear()

                # <<< THAY ĐỔI: Histogram
                # Histogram cho cả 3 logit đầu ra (Hold, Buy, Sell)
                log_payload["val/logit_dist_hold"] = wandb.Histogram(
                    all_logits[:, 0].cpu().numpy()
                )
                log_payload["val/logit_dist_buy"] = wandb.Histogram(
                    all_logits[:, 1].cpu().numpy()
                )
                log_payload["val/logit_dist_sell"] = wandb.Histogram(
                    all_logits[:, 2].cpu().numpy()
                )
                # <<< KẾT THÚC THAY ĐỔI

                # Bảng Per-Coin
                datamodule = self.trainer.datamodule
                all_coin_names = datamodule.coins

                per_coin_ap_metric = MulticlassAveragePrecision(
                    num_classes=3, average=None
                ).to(self.device)

                ap_table_rows = []  # [coin, buy_ap, sell_ap, ap, hold/buy/sell]
                unique_coin_ids = all_coin_ids.unique()

                for coin_id in unique_coin_ids:
                    mask = all_coin_ids == coin_id
                    coin_probs = all_probs[mask]
                    coin_labels = all_labels_multi[mask]
                    if coin_probs.numel() == 0 or coin_labels.numel() == 0:
                        continue

                    coin_aps = per_coin_ap_metric(
                        coin_probs, coin_labels
                    )  # [AP_hold, AP_buy, AP_sell]
                    per_coin_ap_metric.reset()
                    coin_name = all_coin_names[int(coin_id)]

                    buy_ap_coin = coin_aps[1].item()
                    sell_ap_coin = coin_aps[2].item()
                    mean_ap_coin = (buy_ap_coin + sell_ap_coin) / 2.0

                    # Tính tỉ lệ hold/buy/sell cho coin này
                    coin_total = coin_labels.numel()
                    hold_count = (coin_labels == 0).sum().item()
                    buy_count = (coin_labels == 1).sum().item()
                    sell_count = (coin_labels == 2).sum().item()

                    hold_ratio = hold_count / coin_total if coin_total > 0 else 0.0
                    buy_ratio = buy_count / coin_total if coin_total > 0 else 0.0
                    sell_ratio = sell_count / coin_total if coin_total > 0 else 0.0

                    label_dist = f"{hold_ratio:.2f}/{buy_ratio:.2f}/{sell_ratio:.2f}"

                    ap_table_rows.append(
                        [coin_name, buy_ap_coin, sell_ap_coin, mean_ap_coin, label_dist]
                    )

                if ap_table_rows:
                    # Sắp xếp theo mean AP
                    ap_table_rows.sort(key=lambda x: x[3], reverse=True)
                    ap_table = wandb.Table(
                        data=ap_table_rows,
                        columns=["Coin", "Buy AP", "Sell AP", "AP", "Hold/Buy/Sell"],
                    )
                    log_payload["val/ap_table"] = ap_table

                wandb_logger.experiment.log(log_payload)

        # Reset tất cả metrics ở cuối epoch
        self.val_ap.reset()
        self.val_pr_curve.reset()  # <<< THÊM VÀO

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pass

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        """
        (Không thay đổi logic ở đây, ngoại trừ việc logic 'coin_pos_weights'
         sẽ không được sử dụng bởi 'model_step' mới)
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

        # Logic này vẫn chạy nhưng buffer 'coin_pos_weights'
        # không được sử dụng bởi nn.functional.cross_entropy
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
                f"Coin weights calculated (BUT NOT USED by 3-class loss). Min: {pos_weights.min():.2f}, Max: {pos_weights.max():.2f}, Mean: {pos_weights.mean():.2f}"
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
