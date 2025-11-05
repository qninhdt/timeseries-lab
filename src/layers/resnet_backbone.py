import torch
import torch.nn as nn

from .inception_res_block import InceptionResBlock1D


class ResNet1DBackbone(nn.Module):
    def __init__(
        self,
        n_features,
        n_blocks=[2, 2],
        planes_per_branch=[32, 64],
        target_planes=[32, 64],
        num_groups=32,
        context_embed_dim=None,
        dropout=0.1,
    ):
        super().__init__()
        self.base_planes = target_planes[0]
        self.num_groups = num_groups
        self.context_embed_dim = context_embed_dim

        self.stem = nn.Sequential(
            nn.Conv1d(
                n_features,
                self.base_planes,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(self.num_groups, self.base_planes, eps=1e-5),
            nn.ReLU(),
        )
        self.in_planes = self.base_planes

        self.layer1 = self._make_layer(
            planes_per_branch[0],
            target_planes[0],
            n_blocks[0],
            stride=2,
            context_embed_dim=context_embed_dim,
            dropout=dropout,
        )
        self.layer2 = self._make_layer(
            planes_per_branch[1],
            target_planes[1],
            n_blocks[1],
            stride=2,
            context_embed_dim=context_embed_dim,
            dropout=dropout,
        )

        self.out_channels = target_planes[-1]

    def _make_layer(
        self, planes_per_branch, target_planes, num_blocks, stride, context_embed_dim, dropout
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                InceptionResBlock1D(
                    self.in_planes,
                    planes_per_branch,
                    target_planes,
                    context_embed_dim,
                    s,
                    dropout,
                    num_groups=self.num_groups,
                )
            )
            self.in_planes = target_planes
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, context_embedding: torch.Tensor = None) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
        x = self.stem(x)
        for layer in self.layer1:
            x = layer(x, context_embedding)
        for layer in self.layer2:
            x = layer(x, context_embedding)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x
