"""Axis-based cone scores, aperture calibration and attribution loss.

Inputs are image/anchor vectors normalized internally to unit directions.
For an image-axis angle theta and half-aperture psi:
    C = ||image_direction - anchor_direction||^2 = 2*(1-cos(theta))
    W = 2*(1-cos(psi))
    q = C / (W + eps)

Prediction minimizes q. The cone wall is at q=1 up to epsilon regularization.
Scores depend on direction and aperture, not image or anchor radius. Equal
apertures give the same ranking as cosine similarity. Depth conversion helpers
use the spatial anchor norm 2*min_radius/sin(psi).

AxisConeLoss combines six independently weighted terms:
- Center: mean correct-class q, with W detached.
- Coverage: mean hinge outside psi-inside_margin, averaged equally over classes
  present in the batch, with the inner wall detached.
- Aperture: mean log(W) + max(0, q_ap-1)/nu, where q_ap uses the detached
  correct-class angle plus inside_margin, capped at pi.
- Exclusion: mean max(0, 1-q)^2 over selected wrong-class pairs.
- Separation: mean squared angular overlap over violating anchor pairs, including
  separation_margin; zero when no pair violates the constraint.
- Ranking: cross-entropy with logits -q/tau and a learned positive temperature.

Center and coverage update directions; the aperture term updates widths.
Exclusion, separation and ranking can update both. The caller supplies axes and
sin(psi), including any bounds or freezing. nu weights aperture violations;
finite training with other terms and bounded widths does not guarantee coverage.
Calibration returns per-class empirical angle quantiles; the caller checks
allowed widths and pairwise separation before applying them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.attribution_loss import _subsample


def regular_simplex(num_classes: int, dim: int, *, device=None, dtype=None) -> torch.Tensor:
    """Unit vectors with the largest possible common pairwise angle.

    K vertices need at least K-1 dimensions and have pairwise cosine -1/(K-1).
    """
    if num_classes < 2 or dim < num_classes - 1:
        raise ValueError(f"A {num_classes}-vertex simplex needs dim >= {num_classes - 1}")
    centered = (torch.eye(num_classes, device=device, dtype=dtype)
                - torch.full((num_classes, num_classes), 1.0 / num_classes,
                             device=device, dtype=dtype))
    _, basis = torch.linalg.eigh(centered)
    vertices = F.normalize(basis[:, -(num_classes - 1):], dim=-1)
    return F.pad(vertices, (0, dim - (num_classes - 1)))


@torch.no_grad()
def calibrate_axis_apertures(x_img: torch.Tensor, labels: torch.Tensor,
                             x_anc: torch.Tensor, coverage: float,
                             inside_margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-class angular quantiles and achieved padded training coverage.

    Angles include inside_margin in radians. Quantiles use interpolation='higher',
    selecting an observed padded angle rather than interpolating between samples.
    No aperture bounds or pairwise separation constraints are applied here."""
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if x_img.ndim != 2 or x_anc.ndim != 2 or x_img.shape[1] != x_anc.shape[1]:
        raise ValueError("x_img and x_anc must be (N,D) and (K,D) with the same D")
    labels = labels.long()
    if labels.numel() != x_img.shape[0]:
        raise ValueError("labels must have one entry per image")

    dots = (F.normalize(x_img, dim=-1)
            * F.normalize(x_anc, dim=-1)[labels]).sum(-1).clamp(-1.0, 1.0)
    required = torch.arccos(dots) + inside_margin
    psi, covered = [], []
    for k in range(x_anc.shape[0]):
        values = required[labels == k]
        if not len(values):
            raise ValueError(f"Cannot calibrate absent class index {k}")
        wall = torch.quantile(values, coverage, interpolation="higher")
        psi.append(wall)
        covered.append((values <= wall).float().mean())
    return torch.stack(psi), torch.stack(covered)


def cone_wall_chord2(sin_psi: torch.Tensor) -> torch.Tensor:
    """Return squared wall chord radii, 2*(1-cos(psi)), for half-apertures in [0, pi/2].

    Use 2*sin(psi)^2/(1+cos(psi)) to avoid cancellation for narrow cones."""
    sin2 = sin_psi.pow(2)
    cos_psi = (1.0 - sin2).clamp_min(0.0).sqrt()
    return 2.0 * sin2 / (1.0 + cos_psi)


def cone_inner_wall_chord2(sin_psi: torch.Tensor,
                           inside_margin: float) -> torch.Tensor:
    """Return squared chord radii at max(psi-inside_margin, 1e-6), in radians."""
    psi = torch.arcsin(sin_psi.clamp(max=1.0))
    inner_psi = (psi - inside_margin).clamp_min(1e-6)
    return cone_wall_chord2(torch.sin(inner_psi))


def sin_psi_from_depth(x_anc: torch.Tensor, min_radius: float) -> torch.Tensor:
    """Return min(1, 2*min_radius/||anchor||) using Lorentz spatial-coordinate norms."""
    return (2.0 * min_radius / x_anc.norm(dim=-1)).clamp(max=1.0)


def depth_from_sin_psi(sin_psi: torch.Tensor, min_radius: float) -> torch.Tensor:
    """Return spatial anchor norms 2*min_radius/max(sin(psi), 1e-6)."""
    return 2.0 * min_radius / sin_psi.clamp_min(1e-6)


def axis_chord2(x_img: torch.Tensor, x_anc: torch.Tensor) -> torch.Tensor:
    """(B, K) squared chord distance from each cone's AXIS, ‖x̂ − û_k‖² = 2(1 − cos θ)."""
    x_hat = F.normalize(x_img, dim=-1)                              # (B, D)
    u_hat = F.normalize(x_anc, dim=-1)                              # (K, D) axis directions
    # Explicit vector differences avoid cancellation in 1-dot near the axis.
    return (x_hat.unsqueeze(1) - u_hat.unsqueeze(0)).pow(2).sum(-1)


def axis_cone_q(
    x_img: torch.Tensor,          # (B, D) images on the hyperboloid
    x_anc: torch.Tensor,          # (K, D) anchors — only the DIRECTION is read
    sin_psi: torch.Tensor,        # (K,)   sine of each cone's half-aperture
    eps: float = 1e-12,
) -> torch.Tensor:
    """(B, K) squared distance from each cone's axis, in units of that cone's wall."""
    return axis_chord2(x_img, x_anc) / (cone_wall_chord2(sin_psi) + eps)


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor,
                  sin_psi: torch.Tensor) -> torch.Tensor:
    """Image-only inference: the cone whose axis is nearest, in cone-wall units."""
    return axis_cone_q(x_img, x_anc, sin_psi).argmin(dim=1)


class AxisConeLoss(nn.Module):
    def __init__(self, min_radius: float = 0.5, lambda_neg: float = 1.0,
                 neg_samples: int = 0, lambda_aperture: float = 1.0,
                 nu: float = 0.05, lambda_sep: float = 0.0,
                 separation_margin: float = 0.0, inside_margin: float = 0.0,
                 lambda_ce: float = 0.0, ce_tau_init: float = 1.0,
                 lambda_cover: float = 0.0, lambda_center: float = 1.0):
        """Configure objective weights and angular margins.

        Constructor margins are in degrees and stored in radians. neg_samples=0
        uses all wrong classes; positive values subsample negatives per image.
        min_radius only affects the reported equivalent spatial anchor norm.
        Positive lambda_ce creates a learned softplus temperature; anchor and
        aperture tensors are supplied to forward by the caller."""
        super().__init__()
        self.min_radius = min_radius
        self.lambda_neg = lambda_neg
        self.neg_samples = neg_samples
        self.lambda_aperture = lambda_aperture
        self.nu = nu
        self.lambda_sep = lambda_sep
        self.lambda_ce = lambda_ce
        self.lambda_cover = lambda_cover
        self.lambda_center = lambda_center
        self.separation_margin = math.radians(separation_margin)
        self.inside_margin = math.radians(inside_margin)
        if lambda_ce > 0:
            raw = torch.tensor(float(ce_tau_init)).expm1().clamp(min=1e-6).log()
            self.ce_tau_raw = nn.Parameter(raw)

    def forward(
        self,
        x_img: torch.Tensor,       # (B, D)
        x_anc: torch.Tensor,       # (K, D) anchors — only the DIRECTION is read
        labels: torch.Tensor,      # (B,) int in [0, K)
        sin_psi: torch.Tensor,     # (K,) sine of each cone's half-aperture
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted loss and detached batch statistics.

        inside_img is the fraction strictly inside the margin-padded wall.
        viol_mass is mean(q_ap * (q_ap>1)) over all samples, not an outside fraction.
        cone_acc uses argmin q without applying the inside margin or rejection.
        """
        B, K = x_img.shape[0], x_anc.shape[0]
        device = x_img.device

        chord2 = axis_chord2(x_img, x_anc)                           # (B, K)
        wall2 = cone_wall_chord2(sin_psi)                            # (K,)
        q = chord2 / (wall2 + 1e-12)                                 # (B, K)
        pos_idx = labels.unsqueeze(1)
        c_pos = chord2.gather(1, pos_idx).squeeze(1)                 # (B,)
        w_pos = wall2[labels]                                        # (B,)
        q_pos = c_pos / (w_pos + 1e-12)

        # Center gradients reach image/anchor directions with the wall held fixed.
        L_pos = (c_pos / (w_pos.detach() + 1e-12)).mean()
        # Aperture gradients reach widths only; add the inside margin to detached angles.
        cos_pos = (1.0 - 0.5 * c_pos.detach()).clamp(-1.0, 1.0)
        padded_angle = (torch.arccos(cos_pos) + self.inside_margin).clamp(max=math.pi)
        c_cover = 2.0 * (1.0 - torch.cos(padded_angle))
        q_ap = c_cover / (w_pos + 1e-12)
        L_ap = (torch.log(w_pos + 1e-12)
                + (q_ap - 1.0).clamp_min(0.0) / self.nu).mean()

        # Coverage gradients reach directions through chord distances, with the inner
        # wall detached. Each class present in this batch receives equal weight.
        psi = torch.arcsin(sin_psi.clamp(max=1.0))
        inner_wall2 = cone_inner_wall_chord2(sin_psi, self.inside_margin)
        q_inner = c_pos / (inner_wall2[labels].detach() + 1e-12)
        cover_violation = (q_inner - 1.0).clamp_min(0.0)
        present = labels.unique()
        L_cover = torch.stack([
            cover_violation[labels == k].mean() for k in present
        ]).mean()

        neg_mask = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask.scatter_(1, pos_idx, False)
        neg_mask = _subsample(neg_mask, self.neg_samples)
        if neg_mask.any():
            # Wrong-class pairs contribute only when q<1.
            L_neg = (1.0 - q).clamp_min(0.0).pow(2)[neg_mask].mean()
        else:
            L_neg = torch.zeros((), device=device)

        if K > 1:
            d = F.normalize(x_anc, dim=-1)
            cos = (d @ d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            iu = torch.triu_indices(K, K, offset=1, device=device)
            ang = torch.arccos(cos[iu[0], iu[1]])
            need = psi[iu[0]] + psi[iu[1]] + self.separation_margin
            overlap = (need - ang).clamp_min(0.0)
            # Normalize by the number of violating pairs, excluding satisfied pairs.
            active = overlap > 0
            L_sep = (overlap[active].pow(2).mean() if active.any()
                     else torch.zeros((), device=device))
        else:
            ang = torch.full((1,), float("nan"), device=device)
            overlap = torch.zeros(1, device=device)
            L_sep = torch.zeros((), device=device)

        L_ce = torch.zeros((), device=device)
        if self.lambda_ce > 0:
            tau = F.softplus(self.ce_tau_raw)
            L_ce = F.cross_entropy(-q / tau, labels)

        loss = (self.lambda_center * L_pos + self.lambda_cover * L_cover
                + self.lambda_aperture * L_ap + self.lambda_neg * L_neg
                + self.lambda_sep * L_sep + self.lambda_ce * L_ce)

        with torch.no_grad():
            # Report the spatial norm implied by each aperture, not the input anchor norm.
            anc_norm = depth_from_sin_psi(sin_psi, self.min_radius)
            if K > 1:
                sep_min, sep_mean = ang.min(), ang.mean()
            else:
                # Pairwise angle statistics are undefined for a single class.
                sep_min = sep_mean = torch.full((), float("nan"), device=device)
            stats = {
                "loss_pos":     L_pos.detach(),
                "loss_cover":   L_cover.detach(),
                "loss_neg":     L_neg.detach(),
                "loss_ap":      L_ap.detach(),
                "loss_sep":     L_sep.detach(),
                "loss_ce":      L_ce.detach(),
                "ce_tau":       (F.softplus(self.ce_tau_raw).detach()
                                 if self.lambda_ce > 0 else torch.ones((), device=device)),
                "q_pos":        q_pos.mean().detach(),
                "inside_img":   (q_ap < 1.0).float().mean().detach(),
                # Batch mean with zero contribution from samples inside the padded wall.
                "viol_mass":    (q_ap * (q_ap > 1.0).to(q_ap.dtype)).mean().detach(),
                "cone_acc":     (q.argmin(dim=1) == labels).float().mean().detach(),
                # Report the distribution of half-apertures in degrees.
                "psi_min_deg":  torch.rad2deg(psi.min()).detach(),
                "psi_deg":      torch.rad2deg(psi.mean()).detach(),
                "psi_max_deg":  torch.rad2deg(psi.max()).detach(),
                "sep_min_deg":  torch.rad2deg(sep_min).detach(),
                "sep_mean_deg": torch.rad2deg(sep_mean).detach(),
                "sep_overlap":  (overlap > 0).float().mean().detach(),
                "anc_norm":     anc_norm.mean().detach(),
            }
        return loss, stats
