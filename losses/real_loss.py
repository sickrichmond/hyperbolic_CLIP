import torch
import torch.nn as nn


class Step1Loss(nn.Module):
    """
    Loss for the first training step (v1, simple confinement).
    """

    def __init__(
        self,
        r_max: float = 1.0,
        r_min: float = 0.1,
        margin: float = 0.5,
        lambda_push: float = 0.2,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min
        self.margin = margin
        self.lambda_push = lambda_push

    def forward(self, tangent: torch.Tensor, is_real: torch.Tensor):
        dist = tangent.norm(dim=-1)
        real_dist = dist[is_real]
        fake_dist = dist[~is_real]

        if real_dist.numel() > 0:
            outer = torch.clamp(real_dist - self.r_max, min=0.0).pow(2)
            inner = torch.clamp(self.r_min - real_dist, min=0.0).pow(2)
            loss_real = (outer + inner).mean()
        else:
            loss_real = torch.tensor(0.0, device=tangent.device)

        if fake_dist.numel() > 0:
            push_threshold = self.r_max + self.margin
            push = torch.clamp(push_threshold - fake_dist, min=0.0).pow(2)
            loss_push = push.mean()
        else:
            loss_push = torch.tensor(0.0, device=tangent.device)

        total = loss_real + self.lambda_push * loss_push

        return total, {
            "loss_real": loss_real.detach(),
            "loss_push": loss_push.detach(),
            "mean_dist_real": (
                real_dist.mean().detach()
                if real_dist.numel() > 0
                else torch.tensor(0.0, device=tangent.device)
            ),
            "mean_dist_fake": (
                fake_dist.mean().detach()
                if fake_dist.numel() > 0
                else torch.tensor(0.0, device=tangent.device)
            ),
        }