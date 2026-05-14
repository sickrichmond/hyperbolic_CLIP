import torch
import torch.nn as nn


class Step1LossV2(nn.Module):
    """
    Step-1 loss with three terms:

    1. Confinement: keep real samples inside the annulus [r_min, r_max]
       and (weakly) push fake samples above r_max + margin. Same as v1
       but lambda_push is lower because the contrastive term does most
       of the work on fakes now.

    2. Real spread: prevent real samples from collapsing onto a single
       point. Weak repulsion between pairs of real embeddings.

    3. Real-vs-fake contrastive: for each real, the nearest fake in the
       batch (by hyperbolic distance) must be at least margin_contrast
       away. This is the term that forces the model to use angular
       direction in hyperbolic space, not just norm.

    All distances are hyperbolic (acosh-based) so the loss is geometrically
    consistent with the manifold.
    """

    def __init__(
        self,
        # Confinement
        r_max: float = 1.0,
        r_min: float = 0.1,
        margin: float = 0.5,
        lambda_push: float = 0.1,
        # Real spread
        spread_margin: float = 0.2,
        lambda_spread: float = 0.05,
        # Contrastive
        contrast_margin: float = 1.0,
        lambda_contrast: float = 0.5,
        # Hyperbolic
        curv: float = 1.0,
    ):
        super().__init__()
        self.r_max = r_max
        self.r_min = r_min
        self.margin = margin
        self.lambda_push = lambda_push
        self.spread_margin = spread_margin
        self.lambda_spread = lambda_spread
        self.contrast_margin = contrast_margin
        self.lambda_contrast = lambda_contrast
        self.curv = curv
        self.eps = 1e-7

    def _pairwise_hyperbolic_dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise hyperbolic distance between rows of x (space components only).
        x: (B, D)
        returns: (B, B) distances
        """
        x_time = torch.sqrt(1.0 / self.curv + (x ** 2).sum(dim=-1, keepdim=True))
        # Lorentz inner: -t_i t_j + sum(x_i x_j)
        inner = x @ x.T - x_time @ x_time.T
        # -curv * inner should be >= 1
        arg = torch.clamp(-self.curv * inner, min=1.0 + self.eps)
        return torch.acosh(arg) / (self.curv ** 0.5)

    def forward(self, x_hyp: torch.Tensor, tangent: torch.Tensor,
                is_real: torch.Tensor):
        """
        Args:
            x_hyp:   (B, D) space components on the hyperboloid (from exp_map0).
            tangent: (B, D) tangent vectors. Their norm == distance to origin.
            is_real: (B,) bool tensor.
        """
        device = x_hyp.device
        # Distance to origin: simply ||tangent||
        dist0 = tangent.norm(dim=-1)

        # --- 1. Confinement (real annulus + weak fake push) ---
        real_dist = dist0[is_real]
        fake_dist = dist0[~is_real]

        if real_dist.numel() > 0:
            outer = torch.clamp(real_dist - self.r_max, min=0.0).pow(2)
            inner_pen = torch.clamp(self.r_min - real_dist, min=0.0).pow(2)
            loss_real = (outer + inner_pen).mean()
        else:
            loss_real = torch.tensor(0.0, device=device)

        if fake_dist.numel() > 0:
            push_thresh = self.r_max + self.margin
            loss_push = torch.clamp(push_thresh - fake_dist, min=0.0).pow(2).mean()
        else:
            loss_push = torch.tensor(0.0, device=device)

        # --- 2. Real spread: pull real embeddings apart ---
        real_x = x_hyp[is_real]
        if real_x.shape[0] >= 2:
            d_rr = self._pairwise_hyperbolic_dist(real_x)
            # Mask out the diagonal (self-distance is 0)
            mask = ~torch.eye(d_rr.shape[0], dtype=torch.bool, device=device)
            pair_d = d_rr[mask]
            loss_spread = torch.clamp(self.spread_margin - pair_d, min=0.0).pow(2).mean()
        else:
            loss_spread = torch.tensor(0.0, device=device)

        # --- 3. Real-vs-fake contrastive: each real must be far from
        #        its nearest fake in hyperbolic space ---
        if real_x.shape[0] > 0 and (~is_real).sum() > 0:
            fake_x = x_hyp[~is_real]
            # Cross-distance matrix (R, F)
            r_time = torch.sqrt(1.0 / self.curv + (real_x ** 2).sum(dim=-1, keepdim=True))
            f_time = torch.sqrt(1.0 / self.curv + (fake_x ** 2).sum(dim=-1, keepdim=True))
            inner_rf = real_x @ fake_x.T - r_time @ f_time.T
            arg_rf = torch.clamp(-self.curv * inner_rf, min=1.0 + self.eps)
            d_rf = torch.acosh(arg_rf) / (self.curv ** 0.5)  # (R, F)

            # Nearest fake for each real
            nearest_fake_dist, _ = d_rf.min(dim=-1)  # (R,)
            loss_contrast = torch.clamp(
                self.contrast_margin - nearest_fake_dist, min=0.0
            ).pow(2).mean()
        else:
            loss_contrast = torch.tensor(0.0, device=device)

        total = (
            loss_real
            + self.lambda_push * loss_push
            + self.lambda_spread * loss_spread
            + self.lambda_contrast * loss_contrast
        )

        return total, {
            "loss_real": loss_real.detach(),
            "loss_push": loss_push.detach(),
            "loss_spread": loss_spread.detach(),
            "loss_contrast": loss_contrast.detach(),
            "mean_dist_real": real_dist.mean().detach() if real_dist.numel() > 0 else torch.tensor(0.0, device=device),
            "mean_dist_fake": fake_dist.mean().detach() if fake_dist.numel() > 0 else torch.tensor(0.0, device=device),
        }