import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    Paper: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Args:
            dim: Dimension to normalize over
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            Normalized tensor of shape (..., dim)
        """
        # Compute RMS (Root Mean Square)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x * norm * self.weight
