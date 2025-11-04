import torch
import torch.nn as nn
from typing import List, Optional


class Normalize(nn.Module):
    def __init__(
        self,
        num_features: int,
        norm_types: List[int],
        close_idx: int,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        """
        Flexible normalization layer with per-feature normalization strategies.

        :param num_features: the number of features or channels
        :param norm_types: list of normalization types for each feature
                          0 = no normalization
                          1 = normalize with own mean/std
                          2 = normalize with close feature's mean/std
        :param close_idx: index of the close price feature
        :param eps: a value added for numerical stability
        :param affine: if True, has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.norm_types = norm_types
        self.close_idx = close_idx
        self.eps = eps
        self.affine = affine

        assert (
            len(norm_types) == num_features
        ), "norm_types length must match num_features"

        if self.affine:
            self._init_params()

        # Store statistics
        self.mean = None
        self.stdev = None

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize affine params: (num_features,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Calculate mean and std for each feature based on its norm_type.
        x shape: (B, T, D) where B=batch, T=time, D=features
        """
        B, T, D = x.shape

        # Initialize storage for means and stds: (B, 1, D)
        self.mean = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        self.stdev = torch.ones(B, 1, D, device=x.device, dtype=x.dtype)

        for feat_idx in range(D):
            norm_type = self.norm_types[feat_idx]

            if norm_type == 0:
                # No normalization - keep mean=0, std=1
                continue
            elif norm_type == 1:
                # Use own statistics
                feat_data = x[:, :, feat_idx : feat_idx + 1]  # (B, T, 1)
                self.mean[:, :, feat_idx : feat_idx + 1] = torch.mean(
                    feat_data, dim=1, keepdim=True
                )
                self.stdev[:, :, feat_idx : feat_idx + 1] = torch.sqrt(
                    torch.var(feat_data, dim=1, keepdim=True, unbiased=False) + self.eps
                )
            elif norm_type == 2:
                # Use close feature's statistics
                close_data = x[:, :, self.close_idx : self.close_idx + 1]  # (B, T, 1)
                self.mean[:, :, feat_idx : feat_idx + 1] = torch.mean(
                    close_data, dim=1, keepdim=True
                )
                self.stdev[:, :, feat_idx : feat_idx + 1] = torch.sqrt(
                    torch.var(close_data, dim=1, keepdim=True, unbiased=False)
                    + self.eps
                )

    def _normalize(self, x):
        """
        Normalize features based on their norm_types.
        x shape: (B, T, D)
        """
        # Apply normalization per feature
        x_normalized = (x - self.mean) / self.stdev

        # Apply affine transformation if enabled
        if self.affine:
            # affine_weight and affine_bias are (D,), need to broadcast
            x_normalized = x_normalized * self.affine_weight
            x_normalized = x_normalized + self.affine_bias

        # Clip to avoid extreme values
        x_normalized = torch.clamp(x_normalized, -5.0, 5.0)

        return x_normalized

    def _denormalize(self, x):
        """
        Denormalize features back to original scale.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev
        x = x + self.mean

        return x
