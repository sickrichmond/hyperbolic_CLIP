"""TwinNeXt, adapted from OmniDFA upstream commit 35b9052.

https://github.com/teheperinko541/OmniDFA/tree/35b9052e83e05436682095818693493f79da9458/model
"""
import torch
from torch import nn


class TwinNeXt(nn.Module):
    def __init__(self, pretrained=True, out_features=128):
        super().__init__()
        from timm import create_model
        from timm.layers import Mlp

        self.backbone1 = create_model('convnext_small', pretrained=pretrained, num_classes=0)
        self.backbone2 = create_model('convnext_small', pretrained=pretrained, num_classes=0)
        self.mlp = Mlp(self.backbone1.num_features + self.backbone2.num_features,
                       hidden_features=512, out_features=out_features)

    def forward_features(self, local, global_image):
        return torch.cat((self.backbone1(local), self.backbone2(global_image)), dim=1)

    def forward(self, local, global_image):
        return self.mlp(self.forward_features(local, global_image))
