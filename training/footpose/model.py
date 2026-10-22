"""The network: a MobileNetV3 backbone with a small upsampling head.

This follows the "Simple Baselines" pose design (Xiao et al. 2018): take the
backbone's coarsest features and upsample them three times to a heatmap a
quarter of the input size.
"""

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .keypoints import NUM_CHANNELS

# Heatmap pixels per input pixel: 32 from the backbone, divided by 2 three times.
STRIDE = 4


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FootPoseNet(nn.Module):
    def __init__(self, pretrained: bool = True, head_channels: int = 128) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights).features
        # torchvision sets BatchNorm momentum to 0.01 for MobileNetV3, so the
        # running statistics used at inference lag far behind training on a
        # new task. The usual 0.1 keeps them in step.
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.1
        backbone_out = 960
        self.head = nn.Sequential(
            _up_block(backbone_out, head_channels),
            _up_block(head_channels, head_channels),
            _up_block(head_channels, head_channels),
            nn.Conv2d(head_channels, NUM_CHANNELS, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) normalised images to (N, 16, H/4, W/4) heatmaps."""
        return self.head(self.backbone(images))
