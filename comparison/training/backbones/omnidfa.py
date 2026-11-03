"""OmniDFA embedding model and losses, adapted from upstream commit 35b9052.

https://github.com/teheperinko541/OmniDFA/tree/35b9052e83e05436682095818693493f79da9458/model
"""
import torch
from torch import nn
from torch.nn import functional as F


class TwinNeXt(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from timm import create_model
        from timm.layers import Mlp

        self.backbone1 = create_model('convnext_small', pretrained=pretrained, num_classes=0)
        self.backbone2 = create_model('convnext_small', pretrained=pretrained, num_classes=0)
        self.mlp = Mlp(self.backbone1.num_features + self.backbone2.num_features,
                       hidden_features=512, out_features=128)

    def forward(self, local, global_image):
        return self.mlp(torch.cat((self.backbone1(local), self.backbone2(global_image)), dim=1))


def supervised_contrastive_loss(features, labels, temperature=0.07):
    # Float32 keeps the exponential safe under bfloat16 autocast.
    features = F.normalize(features.float(), dim=1)
    similarity = features @ features.T / temperature
    eye = torch.eye(len(labels), device=features.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    denominator = (similarity.exp() * ~eye).sum(dim=1, keepdim=True)
    log_probability = similarity - (denominator + 1e-7).log()
    return (-(positive * log_probability).sum(dim=1) /
            (positive.sum(dim=1) + 1e-7)).mean()


class SphereCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.center_real = nn.Parameter(torch.randn(128))
        self.register_buffer('cosine_threshold', torch.tensor(1.0))

    def forward(self, real_features):
        if not len(real_features):
            raise ValueError('OmniDFA requires real images in every training batch')
        cosine = F.normalize(real_features.float(), dim=1) @ F.normalize(self.center_real, dim=0)
        with torch.no_grad():
            angles = cosine.clamp(-1 + 1e-7, 1 - 1e-7).acos()
            q1, q3 = torch.quantile(angles, angles.new_tensor([0.25, 0.75]))
            valid = angles[angles <= q3 + 1.5 * (q3 - q1)]
            if len(valid):
                self.cosine_threshold.mul_(0.99).add_(valid.max().cos(), alpha=0.01)
                self.cosine_threshold.clamp_(-1 + 1e-7, 1 - 1e-7)
        return (1 - cosine).mean()
