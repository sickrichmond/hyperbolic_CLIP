import torch
import torch.nn as nn

from geometry.lorentz import pairwise_dist


class Step1Loss(nn.Module):
    """
    Step-1 loss with three terms:

      L_real  — confine real samples within the annulus [r_min, r_max].
      L_push  — push fakes beyond r_max + margin (norm-based, weak).
      L_pair  — for each (real, fake) pair sharing the same prompt_id within
                the batch, penalize if their hyperbolic distance is < delta_pair.
                Forces the model to use angular structure on same-prompt pairs,
                where norm cues alone cannot distinguish real from fake.

    L_pair only activates when lambda_pair > 0 AND prompt_id is provided.
    """

    def __init__(
        self,
        r_max: float = 1.0,
        r_min: float = 0.1,
        margin: float = 0.5,
        lambda_push: float = 0.2,
        delta_pair: float = 0.8,
        lambda_pair: float = 0.0,
        curv: float = 1.0,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min
        self.margin = margin
        self.lambda_push = lambda_push
        self.delta_pair = delta_pair
        self.lambda_pair = lambda_pair
        self.curv = curv

    def forward(
        self,
        x_hyp: torch.Tensor,
        tangent: torch.Tensor,
        is_real: torch.Tensor,
        prompt_id: torch.Tensor | None = None,
    ):
        device = tangent.device
        dist0 = tangent.norm(dim=-1)
        real_d = dist0[is_real]
        fake_d = dist0[~is_real]

        # --- L_real ---
        if real_d.numel() > 0:
            outer = torch.clamp(real_d - self.r_max, min=0.0).pow(2)
            inner = torch.clamp(self.r_min - real_d, min=0.0).pow(2)
            loss_real = (outer + inner).mean()
        else:
            loss_real = torch.tensor(0.0, device=device)

        # --- L_push ---
        if fake_d.numel() > 0:
            push_thresh = self.r_max + self.margin
            loss_push = torch.clamp(push_thresh - fake_d, min=0.0).pow(2).mean()
        else:
            loss_push = torch.tensor(0.0, device=device)

        # --- L_pair: per-pair hyperbolic distance between same-prompt (real, fake) ---
        loss_pair = torch.tensor(0.0, device=device)
        if self.lambda_pair > 0 and prompt_id is not None:
            reals = x_hyp[is_real]
            fakes = x_hyp[~is_real]
            if reals.shape[0] > 0 and fakes.shape[0] > 0:
                d_rf = pairwise_dist(reals, fakes, curv=self.curv)   # (R, F)
                reals_pid = prompt_id[is_real]
                fakes_pid = prompt_id[~is_real]
                same_prompt = reals_pid.unsqueeze(1) == fakes_pid.unsqueeze(0)  # (R, F)
                n_pairs = same_prompt.sum()
                if n_pairs > 0:
                    violation = torch.clamp(self.delta_pair - d_rf, min=0.0).pow(2)
                    loss_pair = (violation * same_prompt.float()).sum() / n_pairs

        total = (
            loss_real
            + self.lambda_push * loss_push
            + self.lambda_pair * loss_pair
        )

        return total, {
            "loss_real": loss_real.detach(),
            "loss_push": loss_push.detach(),
            "loss_pair": loss_pair.detach(),
            "mean_dist_real": (
                real_d.mean().detach() if real_d.numel() > 0
                else torch.tensor(0.0, device=device)
            ),
            "mean_dist_fake": (
                fake_d.mean().detach() if fake_d.numel() > 0
                else torch.tensor(0.0, device=device)
            ),
        }
