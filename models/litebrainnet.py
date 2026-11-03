import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LiteBrainNet(nn.Module):
    """Compact four-class CNN; its outputs are not meaningful before training."""

    def __init__(self, num_classes=4, input_channels=1, dropout=0.25, channels=(16, 32, 64, 128)):
        super().__init__()
        if len(channels) != 4 or min(channels) < 1:
            raise ValueError("channels must contain four positive widths")
        c1, c2, c3, c4 = map(int, channels)
        self.features = nn.Sequential(
            ConvBlock(input_channels, c1, stride=2, dropout=dropout / 2),
            DepthwiseSeparableBlock(c1, c2),
            DepthwiseSeparableBlock(c2, c3),
            DepthwiseSeparableBlock(c3, c4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c4, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

    @property
    def gradcam_layer(self):
        return self.features[-1]


def count_parameters(model, trainable_only=True):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
