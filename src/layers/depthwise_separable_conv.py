import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_planes,
            in_planes,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

