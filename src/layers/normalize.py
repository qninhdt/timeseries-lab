import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class Normalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        norm_types: List[int],
        close_idx: int,
        price_feature_indices: Optional[List[int]] = None,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        """
        Multi-timeframe normalization layer.
        Price features across all timeframes use 1h close statistics.
        Other features use their own statistics per timeframe.

        :param num_features: number of features for 1h (max features)
        :param norm_types: normalization type per feature (0=no norm, 1=own stats, 2=close stats)
        :param close_idx: index of close price feature
        :param price_feature_indices: indices of price features (open, high, low, close, volume)
        :param eps: numerical stability epsilon
        :param affine: enable learnable affine transformation (separate for each timeframe)
        """
        super().__init__()
        self.num_features = num_features
        self.norm_types = norm_types
        self.close_idx = close_idx
        self.eps = eps
        self.affine = affine

        assert (
            len(norm_types) == num_features
        ), "norm_types length must match num_features"

        self.price_feature_indices = (
            price_feature_indices
            if price_feature_indices is not None
            else [i for i, nt in enumerate(norm_types) if nt == 2]
        )

        # Initialize affine parameters for each timeframe (will be sliced based on actual feature count)
        if self.affine:
            # Initialize with max features (1h), will slice for 4h/1d
            self.affine_weight_1h = nn.Parameter(torch.ones(num_features))
            self.affine_bias_1h = nn.Parameter(torch.zeros(num_features))
            self.affine_weight_4h = nn.Parameter(torch.ones(num_features))
            self.affine_bias_4h = nn.Parameter(torch.zeros(num_features))
            self.affine_weight_1d = nn.Parameter(torch.ones(num_features))
            self.affine_bias_1d = nn.Parameter(torch.zeros(num_features))

        # Statistics storage (set in _get_statistics)
        self.mean_1h = None
        self.stdev_1h = None
        self.mean_4h = None
        self.stdev_4h = None
        self.mean_1d = None
        self.stdev_1d = None

    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor,
        mode: str = "norm",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize three timeframes.

        Args:
            x_1h: (B, T_1h, D_1h) - 1h timeframe features
            x_4h: (B, T_4h, D_4h) - 4h timeframe features (common features only)
            x_1d: (B, T_1d, D_1d) - 1d timeframe features (common features only)
            mode: "norm" (denorm removed)

        Returns:
            Tuple of (x_1h_norm, x_4h_norm, x_1d_norm)
        """
        if mode != "norm":
            raise ValueError(f"Only mode='norm' is supported, got {mode}")

        self._get_statistics(x_1h, x_4h, x_1d)
        x_1h_norm = self._normalize(x_1h, self.mean_1h, self.stdev_1h, "1h")
        x_4h_norm = self._normalize(x_4h, self.mean_4h, self.stdev_4h, "4h")
        x_1d_norm = self._normalize(x_1d, self.mean_1d, self.stdev_1d, "1d")
        return x_1h_norm, x_4h_norm, x_1d_norm

    def _get_statistics(
        self, x_1h: torch.Tensor, x_4h: torch.Tensor, x_1d: torch.Tensor
    ) -> None:
        """Calculate mean and std for each feature per timeframe."""
        B_1h, T_1h, D_1h = x_1h.shape
        B_4h, T_4h, D_4h = x_4h.shape
        B_1d, T_1d, D_1d = x_1d.shape
        D_common = min(D_4h, D_1d)

        # Initialize statistics
        self.mean_1h = torch.zeros(B_1h, 1, D_1h, device=x_1h.device, dtype=x_1h.dtype)
        self.stdev_1h = torch.ones(B_1h, 1, D_1h, device=x_1h.device, dtype=x_1h.dtype)
        self.mean_4h = torch.zeros(B_4h, 1, D_4h, device=x_4h.device, dtype=x_4h.dtype)
        self.stdev_4h = torch.ones(B_4h, 1, D_4h, device=x_4h.device, dtype=x_4h.dtype)
        self.mean_1d = torch.zeros(B_1d, 1, D_1d, device=x_1d.device, dtype=x_1d.dtype)
        self.stdev_1d = torch.ones(B_1d, 1, D_1d, device=x_1d.device, dtype=x_1d.dtype)

        # Calculate 1h close statistics (used for price features across all timeframes)
        close_1h = x_1h[:, :, self.close_idx : self.close_idx + 1]  # (B, T, 1)
        close_1h_mean = torch.mean(close_1h, dim=1, keepdim=True)  # (B, 1, 1)
        close_1h_std = torch.sqrt(
            torch.var(close_1h, dim=1, keepdim=True, unbiased=False) + self.eps
        )  # (B, 1, 1)

        # Process common features (present in all timeframes)
        for feat_idx in range(D_common):
            norm_type = self.norm_types[feat_idx]
            is_price = feat_idx in self.price_feature_indices

            if norm_type == 0:
                continue
            elif norm_type == 1 and not is_price:
                # Use own statistics per timeframe
                self._set_stats(
                    x_1h[:, :, feat_idx : feat_idx + 1],
                    self.mean_1h,
                    self.stdev_1h,
                    feat_idx,
                )
                self._set_stats(
                    x_4h[:, :, feat_idx : feat_idx + 1],
                    self.mean_4h,
                    self.stdev_4h,
                    feat_idx,
                )
                self._set_stats(
                    x_1d[:, :, feat_idx : feat_idx + 1],
                    self.mean_1d,
                    self.stdev_1d,
                    feat_idx,
                )
            elif norm_type == 2 or is_price:
                # Price features: use 1h close statistics for all timeframes
                self.mean_1h[:, :, feat_idx : feat_idx + 1] = close_1h_mean
                self.stdev_1h[:, :, feat_idx : feat_idx + 1] = close_1h_std
                self.mean_4h[:, :, feat_idx : feat_idx + 1] = close_1h_mean
                self.stdev_4h[:, :, feat_idx : feat_idx + 1] = close_1h_std
                self.mean_1d[:, :, feat_idx : feat_idx + 1] = close_1h_mean
                self.stdev_1d[:, :, feat_idx : feat_idx + 1] = close_1h_std

        # Process 1h-only features
        for feat_idx in range(D_common, D_1h):
            norm_type = self.norm_types[feat_idx]
            if norm_type == 0:
                continue
            elif norm_type == 1:
                self._set_stats(
                    x_1h[:, :, feat_idx : feat_idx + 1],
                    self.mean_1h,
                    self.stdev_1h,
                    feat_idx,
                )
            elif norm_type == 2:
                self.mean_1h[:, :, feat_idx : feat_idx + 1] = close_1h_mean
                self.stdev_1h[:, :, feat_idx : feat_idx + 1] = close_1h_std

    def _set_stats(
        self, feat: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, feat_idx: int
    ) -> None:
        """Set mean and std for a single feature."""
        mean[:, :, feat_idx : feat_idx + 1] = torch.mean(feat, dim=1, keepdim=True)
        stdev[:, :, feat_idx : feat_idx + 1] = torch.sqrt(
            torch.var(feat, dim=1, keepdim=True, unbiased=False) + self.eps
        )

    def _normalize(
        self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, timeframe: str
    ) -> torch.Tensor:
        """Normalize features for a specific timeframe."""
        x_norm = (x - mean) / stdev

        if self.affine:
            D = x.shape[-1]
            if timeframe == "1h":
                weight = self.affine_weight_1h[:D]
                bias = self.affine_bias_1h[:D]
            elif timeframe == "4h":
                weight = self.affine_weight_4h[:D]
                bias = self.affine_bias_4h[:D]
            elif timeframe == "1d":
                weight = self.affine_weight_1d[:D]
                bias = self.affine_bias_1d[:D]
            else:
                raise ValueError(f"Unknown timeframe: {timeframe}")
            x_norm = x_norm * weight + bias

        return torch.clamp(x_norm, -5.0, 5.0)
