import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.body(x) + self.skip(x))


class ResidualUNet(nn.Module):
    """Small residual U-Net baseline; requires a defensible training target."""

    def __init__(self, input_channels=1, output_channels=1, channels=(16, 32, 64, 128)):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels must contain at least two widths")
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = input_channels
        for width in channels:
            self.encoders.append(ResidualBlock(in_ch, width))
            self.downsamples.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
            in_ch = width
        self.bottleneck = ResidualBlock(channels[-1], channels[-1] * 2)
        self.decoders = nn.ModuleList()
        current = channels[-1] * 2
        for width in reversed(channels):
            self.decoders.append(ResidualBlock(current + width, width))
            current = width
        self.output = nn.Conv2d(channels[0], output_channels, 1)

    def forward(self, x):
        skips = []
        current = x
        for encoder, downsample in zip(self.encoders, self.downsamples):
            current = encoder(current)
            skips.append(current)
            current = downsample(current)
        current = self.bottleneck(current)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = decoder(torch.cat((current, skip), dim=1))
        return self.output(current)
