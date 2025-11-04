"""
PyTorch implementations of time series augmentation techniques.
All functions operate directly on GPU tensors without CPU/numpy conversion.
"""

import torch


def jitter(
    x: torch.Tensor, sigma: float = 0.03, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Add Gaussian noise to time series.

    Args:
        x: input tensor (batch, time, features)
        sigma: standard deviation of noise
        mask: boolean mask (features,) indicating which features to augment

    Returns:
        augmented tensor with same shape as input
    """
    noise = torch.randn_like(x) * sigma

    if mask is not None:
        # Broadcast mask to (1, 1, features)
        mask_expanded = mask.view(1, 1, -1)
        noise = noise * mask_expanded

    return x + noise


def scaling(
    x: torch.Tensor, sigma: float = 0.1, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Scale time series by random factors.

    Args:
        x: input tensor (batch, time, features)
        sigma: standard deviation of scaling factor
        mask: boolean mask (features,) indicating which features to augment

    Returns:
        augmented tensor with same shape as input
    """
    batch_size, time_steps, num_features = x.shape

    # Generate scaling factors: (batch, features)
    factor = (
        torch.randn(batch_size, num_features, device=x.device, dtype=x.dtype) * sigma
        + 1.0
    )

    if mask is not None:
        # Only apply scaling to masked features, keep others at 1.0
        factor = torch.where(mask.unsqueeze(0), factor, torch.ones_like(factor))

    # Expand to (batch, 1, features) and multiply
    factor_expanded = factor.unsqueeze(1)
    return x * factor_expanded


def apply_augmentation(
    x: torch.Tensor,
    feature_mask: torch.Tensor,
    jitter_prob: float = 0.5,
    scaling_prob: float = 0.5,
    jitter_sigma: float = 0.03,
    scaling_sigma: float = 0.1,
) -> torch.Tensor:
    """
    Apply multiple augmentations with probabilities.
    Only augments features where feature_mask is True (norm_type=2).

    Args:
        x: input tensor (batch, time, features)
        feature_mask: boolean mask (features,) indicating which features to augment
        jitter_prob: probability of applying jitter
        scaling_prob: probability of applying scaling
        jitter_sigma: sigma for jitter augmentation
        scaling_sigma: sigma for scaling augmentation

    Returns:
        augmented tensor
    """
    x_aug = x

    # Apply jitter with probability
    if torch.rand(1).item() < jitter_prob:
        x_aug = jitter(x_aug, sigma=jitter_sigma, mask=feature_mask)

    # Apply scaling with probability
    if torch.rand(1).item() < scaling_prob:
        x_aug = scaling(x_aug, sigma=scaling_sigma, mask=feature_mask)

    return x_aug
