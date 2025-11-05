import torch
import torch.nn as nn

from .depthwise_separable_conv import DepthwiseSeparableConv1d


class InceptionResBlock1D(nn.Module):
    def __init__(
        self,
        in_planes,
        planes_per_branch,
        target_planes,
        context_embed_dim=None,
        stride=1,
        dropout=0.1,
        num_groups=32,
    ):
        super().__init__()
        self.stride = stride
        self.out_planes = target_planes
        self.in_planes = in_planes
        self.concat_planes = planes_per_branch * 3

        self.branch_k3 = DepthwiseSeparableConv1d(
            in_planes,
            planes_per_branch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.branch_k5 = DepthwiseSeparableConv1d(
            in_planes,
            planes_per_branch,
            kernel_size=5,
            stride=stride,
            padding=2,
            bias=False,
        )
        self.branch_k7 = DepthwiseSeparableConv1d(
            in_planes,
            planes_per_branch,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(num_groups, self.concat_planes, eps=1e-4)
        self.silu = nn.SiLU(inplace=True)

        self.conv_1x1_out = nn.Conv1d(
            self.concat_planes,
            target_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups, target_planes, eps=1e-4)

        # FiLM generator (only if context_embed_dim is provided)
        if context_embed_dim is not None:
            self.film_generator = nn.Sequential(
                nn.Linear(context_embed_dim, self.concat_planes * 2),
            )
        else:
            self.film_generator = None

        self.downsample = None
        if stride > 1 or in_planes != target_planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_planes, target_planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(num_groups, target_planes, eps=1e-4),
            )

    def forward(
        self, x: torch.Tensor, context_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # layer 1
        out_k3 = self.branch_k3(x)
        out_k5 = self.branch_k5(x)
        out_k7 = self.branch_k7(x)
        out = torch.cat([out_k3, out_k5, out_k7], dim=1)
        out = self.norm1(out)
        out = self.silu(out)

        # FiLM (apply only if context_embedding and film_generator are available)
        if context_embedding is not None and self.film_generator is not None:
            gamma_beta = self.film_generator(context_embedding)
            gamma, beta = torch.split(gamma_beta, self.concat_planes, dim=1)
            out = (gamma.unsqueeze(2) * out) + beta.unsqueeze(2)

        # layer 2
        out = self.conv_1x1_out(out)
        out = self.norm2(out)
        out = self.silu(out)

        out += identity

        return out
