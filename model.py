import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class DSConv(nn.Module):
    """Depthwise-separable convolution block."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                cin, cin,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=cin,
                bias=False,
            ),
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),

            nn.Conv2d(cin, cout, kernel_size=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """Two standard 3x3 convolutions."""
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),

            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualDSConv(nn.Module):
    """Residual block based on depthwise-separable convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            DSConv(channels, channels),
            DSConv(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualConvBlock(nn.Module):
    """Residual block with standard 3x3 convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------
# V1 - original lightweight network
# ---------------------------------------------------------------------

class TinyEdgeNet(nn.Module):
    """
    Original TinyEdgeNet (V1).

    Input:
        B x 3 x H x W

    Output:
        B x 1 x H x W logits
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        self.enc1 = DSConv(24, 32, stride=2)
        self.enc2 = DSConv(32, 64, stride=2)
        self.enc3 = DSConv(64, 96, stride=2)

        self.mid = nn.Sequential(
            DSConv(96, 96),
            DSConv(96, 96),
        )

        self.dec2 = DSConv(96 + 64, 64)
        self.dec1 = DSConv(64 + 32, 32)
        self.dec0 = DSConv(32 + 24, 24)

        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        z = self.mid(s3)

        z = F.interpolate(
            z, size=s2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec2(torch.cat([z, s2], dim=1))

        z = F.interpolate(
            z, size=s1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec1(torch.cat([z, s1], dim=1))

        z = F.interpolate(
            z, size=s0.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec0(torch.cat([z, s0], dim=1))

        return self.head(z)


# ---------------------------------------------------------------------
# V2 - wider encoder, residual bottleneck, stronger decoder
# ---------------------------------------------------------------------

class TinyEdgeNetV2(nn.Module):
    """
    TinyEdgeNet V2.

    Wider than V1 and with:
    - standard-convolution stem
    - residual bottleneck
    - standard-convolution decoder
    - high-resolution refinement
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.stem = ConvBlock(in_channels, 32)

        self.enc1 = DSConv(32, 48, stride=2)
        self.enc2 = DSConv(48, 96, stride=2)
        self.enc3 = DSConv(96, 160, stride=2)

        self.mid = nn.Sequential(
            ResidualDSConv(160),
            ResidualDSConv(160),
        )

        self.dec2 = ConvBlock(160 + 96, 96)
        self.dec1 = ConvBlock(96 + 48, 48)
        self.dec0 = ConvBlock(48 + 32, 32)

        self.refine = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)

        z = self.mid(s3)

        z = F.interpolate(
            z, size=s2.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec2(torch.cat([z, s2], dim=1))

        z = F.interpolate(
            z, size=s1.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec1(torch.cat([z, s1], dim=1))

        z = F.interpolate(
            z, size=s0.shape[-2:],
            mode="bilinear", align_corners=False
        )
        z = self.dec0(torch.cat([z, s0], dim=1))

        z = self.refine(z)
        return self.head(z)


# ---------------------------------------------------------------------
# V3 - stronger multi-scale network, same single-logit output interface
# ---------------------------------------------------------------------

class TinyEdgeNetV3(nn.Module):
    """
    TinyEdgeNet V3.

    Main changes vs V2:
    - wider channels: 32 -> 64 -> 128 -> 192
    - residual refinement at each encoder scale
    - deeper residual bottleneck
    - stronger decoder
    - multi-scale side edge heads
    - learned fusion of final + side predictions

    IMPORTANT:
    forward() still returns a single B x 1 x H x W logits tensor.
    Therefore the existing training loop/loss can remain unchanged.
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # Full resolution
        self.stem = ConvBlock(in_channels, 32)

        # H/2
        self.enc1_down = DSConv(32, 64, stride=2)
        self.enc1_refine = ResidualDSConv(64)

        # H/4
        self.enc2_down = DSConv(64, 128, stride=2)
        self.enc2_refine = ResidualDSConv(128)

        # H/8
        self.enc3_down = DSConv(128, 192, stride=2)
        self.enc3_refine = ResidualDSConv(192)

        # Deeper bottleneck, but no additional downsampling
        self.mid = nn.Sequential(
            ResidualConvBlock(192),
            ResidualConvBlock(192),
            ResidualConvBlock(192),
        )

        # Decoder
        self.dec2 = ConvBlock(192 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.dec0 = ConvBlock(64 + 32, 32)

        self.refine = nn.Sequential(
            ResidualConvBlock(32),
            nn.Conv2d(32, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

        # Main high-resolution prediction
        self.main_head = nn.Conv2d(24, 1, 1)

        # Multi-scale side predictions
        self.side1 = nn.Conv2d(64, 1, 1)    # H/2
        self.side2 = nn.Conv2d(128, 1, 1)   # H/4
        self.side3 = nn.Conv2d(192, 1, 1)   # H/8

        # Fuse [main, side1, side2, side3] logits
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, bias=True)

    def forward(self, x):
        input_size = x.shape[-2:]

        s0 = self.stem(x)

        s1 = self.enc1_down(s0)
        s1 = self.enc1_refine(s1)

        s2 = self.enc2_down(s1)
        s2 = self.enc2_refine(s2)

        s3 = self.enc3_down(s2)
        s3 = self.enc3_refine(s3)

        z = self.mid(s3)

        z = F.interpolate(
            z,
            size=s2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z = self.dec2(torch.cat([z, s2], dim=1))

        z = F.interpolate(
            z,
            size=s1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z = self.dec1(torch.cat([z, s1], dim=1))

        z = F.interpolate(
            z,
            size=s0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        z = self.dec0(torch.cat([z, s0], dim=1))

        z = self.refine(z)
        main = self.main_head(z)

        # Side logits at original resolution
        side1 = F.interpolate(
            self.side1(s1),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        side2 = F.interpolate(
            self.side2(s2),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        side3 = F.interpolate(
            self.side3(s3),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        fused = self.fuse(
            torch.cat([main, side1, side2, side3], dim=1)
        )

        return fused
