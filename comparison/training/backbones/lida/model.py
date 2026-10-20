import warnings

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class LIDABackbone(nn.Module):
    """LIDA's modified ResNet-50 feature extractor.

    LIDA removes the first max-pool and the first stride in layer1 so that
    low-bit forensic evidence is retained at a higher spatial resolution.
    """

    output_dim = 2048

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
        except Exception:
            if pretrained:
                warnings.warn(
                    "Pretrained ResNet50 weights could not be loaded (offline/network restricted). "
                    "Falling back to random initialization.",
                    RuntimeWarning,
                )
            backbone = resnet50(weights=None)

        backbone.maxpool = nn.Identity()
        backbone.layer1[0].conv1.stride = (1, 1)
        backbone.layer1[0].downsample[0].stride = (1, 1)

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

    def forward(self, images):
        return torch.flatten(self.features(images), 1)