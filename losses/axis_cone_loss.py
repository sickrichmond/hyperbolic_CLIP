"""Axis-distance cone loss for attribution.

One idea: an image should sit ON THE AXIS of its class's cone, and OUTSIDE every other
class's cone. Nothing else.

Geometry (Lorentz model, curvature c=1, space components only — see geometry/lorentz.py).
Write K = min_radius, a_k for the class anchor on the hyperboloid, x for an image.

  axis        The cone of apex a_k is bisected by the geodesic through a_k. exp_map0 is
              radial, so in space components that geodesic is the straight ray of
              direction û_k = a_k/‖a_k‖.

  distance    Distance from the axis, as a chord between directions:

                  δ(x, axis_k)² = ‖x̂ − û_k‖² = 2(1 − cos θ)

              and the same quantity evaluated ON the cone wall, where θ = ψ_k:

                  δ_wall(k)²    = 2(1 − cos ψ_k)

  aperture    sin ψ_k, a FREE per-class parameter — see "Why psi is its own parameter".
              The equivalent depth is ‖a_k‖ = 2K/sin ψ_k, so the anchor is still a point
              in hyperbolic space and the aperture is still tied to depth; only the
              parameterisation differs.

  score       The squared distance from the axis, in units of the squared distance at
              the wall — a ratio of two distances, dimensionless:

                            δ(x, axis_k)²      1 − cos θ         c_ik
                    q_ik =  ─────────────  =  ───────────  =  ─────────
                             δ_wall(k)²        1 − cos ψ_k        W_k

              q = 0 on the axis · q = 1 exactly on the cone WALL · q > 1 outside.

Loss:
    L = mean_i [ q_{i,y_i} + λ_ap · log W_{y_i} + λ_neg · mean_{k ∈ N_i} max(0, 1 − q_ik)² ]
    subject to ‖u_k‖ ∈ [r_min, r_max]  (a hard projection in the trainer, not a penalty)

The positive term is the squared normalised distance from the right axis: zero only ON
the axis, gradient everywhere, never saturating. The negative term says "stay out of
other people's cones" and needs NO margin hyper-parameter — the margin is the cone wall.

The APERTURE term is what sets ψ, and it is not optional. Without it the only pressure on
the aperture is one-way: q = c/W is lowered by growing W, so every anchor walks to the
smallest radius the range allows and ψ ends up uniform — measured, ‖t_anc‖ pinned at the
floor to the digit and ψ ∈ [28.0, 28.0]. The negative term cannot supply the counterforce:
with anchors ~68° apart and cones ~28° wide, an image inside its own cone is 40° from
every other axis, so max(0, 1 − q_neg) is exactly zero and stays there. With ψ uniform,
argmin q IS argmax cos and the whole construction is a cosine again.

log W restores the second direction and costs no hyper-parameter search, because its
equilibrium is exact and readable:

    ∂/∂W [ mean_i c_ik/W + λ_ap log W ] = 0   ⇒   W_k = mean_i(c_ik) / λ_ap

i.e. at λ_ap = 1 the cone wall sits exactly on the RMS angular radius of its own class,
and mean_i q_i,k = 1. Tight classes get narrow cones, diffuse ones get wide cones — so ψ
varies across classes for a reason that comes from the data, which is the necessary
condition for argmin q to differ from argmax cos at all.

Three properties this form has and the obvious alternatives do not:

  MONOTONE on all of [0, π].  The natural-looking (sinh d_axis / sinh R)² = (sinθ/sinψ)²
  is BILATERAL — ‖x⊥‖ is invariant under x → −x — so its derivative flips sign at 90°
  and beyond that, lowering it means walking toward 180°. That is an antipodal attractor,
  and it is exactly the degeneracy Run C hit with a pair of anchors at 179 degrees. The
  one-sided ray distance is monotone but goes FLAT past 90°, which is a dead gradient
  zone right where random anchors start. The chord ratio has neither problem.

  ALGEBRAIC.  No arccos, acosh, asin or asinh anywhere, hence no clamp for a gradient to
  die on — the failure that stopped the two previous formulations. ∂q/∂cos θ =
  −1/(1 − cos ψ) is constant.

  DEPTH-INVARIANT.  q does not involve ‖x‖, so there is no pressure to collapse images
  toward the origin, and no need to calibrate how deep the projection head starts.

Why psi is its own parameter, and not 2K/‖a_k‖ as the entailment-cone formula has it.

Deriving the aperture from the anchor's depth makes "widen my cone" and "move toward the
origin" THE SAME ACTION. That hands the optimiser a scalar knob which lowers q for every
sample of the class at once, costs nothing, and which nothing opposes — while the
alternative, rotating the axis toward the class, is a 128-dimensional move that has to
compete with 21 other anchors. It takes the knob every time. Measured over five epochs:
psi 53.3° → 65.0° monotone, ‖t_anc‖ down to the norm range's floor and pinned there, the
psi SPREAD collapsing 8.7° → 0.6° — and a uniform psi is exactly the condition under which
argmin q IS argmax cos.

Decoupled, u_k carries only the direction and psi_k only the aperture, so neither can
substitute for the other:
    direction (tangential): via cos θ = ⟨x̂, û_k⟩. The axis of the correct class turns
        TOWARD the image; the axis of a wrong class whose cone swallowed it turns AWAY.
        The radial component of the gradient on u_k is exactly zero — q reads a normalised
        direction — so the trainer is free to keep ‖u_k‖ slaved to psi for display.
    aperture: the positive term pushes psi UP (widen until you contain your own class),
        the aperture term pushes it DOWN (its equilibrium is below), the negative term
        pushes it down too when a wrong image is inside. Bounded by construction.

Inference:  ŷ = argmin_k q_ik.  Open-set: reject when min_k q_ik > 1 (outside every cone).
Since the ψ_k differ across classes, argmin q is NOT argmax cos — which is the point:
with equal ψ the two are the same rule algebraically, and that is what pinned every
earlier run to a 0.9985-0.9998 cone-cosine agreement.

What this is NOT: q = (1 − cos θ)/(1 − cos ψ_k) is a cosine classifier with a per-class
learned angular scale. The hyperbolic content is the aperture-depth relation
ψ = asin(2K/‖a‖) — specificity IS depth, which is what buys nested cones and the
parameter-free open-set rule. Claiming more than that would be wrong.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.attribution_loss import _subsample


def cone_wall_chord2(sin_psi: torch.Tensor) -> torch.Tensor:
    """(K,) squared chord distance from the axis to the cone wall, 2(1 − cos ψ).

    Written as sin²ψ/(1+cos ψ) rather than 1 − cos ψ: for a narrow cone cos ψ is within
    1e-4 of 1 and the subtraction would lose most of its significant digits.
    """
    sin2 = sin_psi.pow(2)
    cos_psi = (1.0 - sin2).clamp_min(0.0).sqrt()
    return 2.0 * sin2 / (1.0 + cos_psi)


def sin_psi_from_depth(x_anc: torch.Tensor, min_radius: float) -> torch.Tensor:
    """(K,) the COUPLED parameterisation, sin ψ = min(1, 2K/‖a‖). Kept for checkpoints
    written before psi became a free parameter, and to convert a psi back to a depth."""
    return (2.0 * min_radius / x_anc.norm(dim=-1)).clamp(max=1.0)


def depth_from_sin_psi(sin_psi: torch.Tensor, min_radius: float) -> torch.Tensor:
    """(K,) hyperboloid norm ‖a‖ = 2K/sin ψ. The anchor is still a point in hyperbolic
    space at the depth its aperture implies; only the parameterisation changed."""
    return 2.0 * min_radius / sin_psi.clamp_min(1e-6)


def axis_cone_q(
    x_img: torch.Tensor,          # (B, D) images on the hyperboloid
    x_anc: torch.Tensor,          # (K, D) anchors — only the DIRECTION is read
    sin_psi: torch.Tensor,        # (K,)   sine of each cone's half-aperture
    eps: float = 1e-12,
) -> torch.Tensor:
    """(B, K) squared distance from each cone's axis, in units of that cone's wall."""
    x_hat = F.normalize(x_img, dim=-1)                              # (B, D)
    u_hat = F.normalize(x_anc, dim=-1)                              # (K, D) axis directions
    # ‖x̂ − û‖², from the vector difference rather than 2(1 − x̂·û): the subtraction form
    # cancels catastrophically exactly where it matters, near the axis.
    chord2 = (x_hat.unsqueeze(1) - u_hat.unsqueeze(0)).pow(2).sum(-1)   # (B, K)
    return chord2 / (cone_wall_chord2(sin_psi) + eps)


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor,
                  sin_psi: torch.Tensor) -> torch.Tensor:
    """Image-only inference: the cone whose axis is nearest, in cone-wall units."""
    return axis_cone_q(x_img, x_anc, sin_psi).argmin(dim=1)


class AxisConeLoss(nn.Module):
    def __init__(self, min_radius: float = 0.5, lambda_neg: float = 1.0,
                 neg_samples: int = 0, lambda_aperture: float = 1.0):
        """neg_samples = 0 uses all K-1 negatives; k > 0 keeps k at random per sample.
        lambda_aperture = 0 removes the log-W term, so nothing pushes the aperture down.
        min_radius is only used to report the equivalent depth. This module has no
        parameters; the anchors and their apertures belong to the trainer."""
        super().__init__()
        self.min_radius = min_radius
        self.lambda_neg = lambda_neg
        self.neg_samples = neg_samples
        self.lambda_aperture = lambda_aperture

    def forward(
        self,
        x_img: torch.Tensor,       # (B, D)
        x_anc: torch.Tensor,       # (K, D) anchors — only the DIRECTION is read
        labels: torch.Tensor,      # (B,) int in [0, K)
        sin_psi: torch.Tensor,     # (K,) sine of each cone's half-aperture
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, K = x_img.shape[0], x_anc.shape[0]
        device = x_img.device

        q = axis_cone_q(x_img, x_anc, sin_psi)                       # (B, K)
        pos_idx = labels.unsqueeze(1)
        q_pos = q.gather(1, pos_idx).squeeze(1)                      # (B,)
        L_pos = q_pos.mean()

        # Aperture: log of the wall for each sample's OWN class, so every class balances
        # against its own spread. Batch-frequency weighting cancels in the ratio, which
        # is why the equilibrium mean_i q = lambda_aperture is exact per class.
        wall2 = cone_wall_chord2(sin_psi)                            # (K,)
        L_ap = torch.log(wall2[labels] + 1e-12).mean()

        neg_mask = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask.scatter_(1, pos_idx, False)
        neg_mask = _subsample(neg_mask, self.neg_samples)
        if neg_mask.any():
            # Only cones that actually contain a wrong image are penalised; already
            # outside (q > 1) contributes nothing. That is the one saturation we want,
            # and it is on a constraint that is genuinely satisfied.
            L_neg = (1.0 - q).clamp_min(0.0).pow(2)[neg_mask].mean()
        else:
            L_neg = torch.zeros((), device=device)

        loss = L_pos + self.lambda_aperture * L_ap + self.lambda_neg * L_neg

        with torch.no_grad():
            psi = torch.arcsin(sin_psi.clamp(max=1.0))
            # the depth the aperture implies — the anchor's position in hyperbolic space,
            # which is no longer what the parameter stores
            anc_norm = depth_from_sin_psi(sin_psi, self.min_radius)
            if K > 1:
                d = F.normalize(x_anc, dim=-1)
                cos = (d @ d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                iu = torch.triu_indices(K, K, offset=1, device=device)
                ang = torch.arccos(cos[iu[0], iu[1]])
                sep_min, sep_mean = ang.min(), ang.mean()
            else:
                # No pairs to separate. NaN rather than a made-up angle, so a run with
                # one class shows up as unreadable instead of quietly plausible.
                sep_min = sep_mean = torch.full((), float("nan"), device=device)
            stats = {
                "loss_pos":     L_pos.detach(),
                "loss_neg":     L_neg.detach(),
                "loss_ap":      L_ap.detach(),
                "q_pos":        q_pos.mean().detach(),
                "inside_img":   (q_pos < 1.0).float().mean().detach(),
                "cone_acc":     (q.argmin(dim=1) == labels).float().mean().detach(),
                # psi SPREAD is the necessary condition for argmin q to differ from
                # argmax cos at all. If min and max stay equal, we rebuilt a cosine.
                "psi_min_deg":  torch.rad2deg(psi.min()).detach(),
                "psi_deg":      torch.rad2deg(psi.mean()).detach(),
                "psi_max_deg":  torch.rad2deg(psi.max()).detach(),
                "sep_min_deg":  torch.rad2deg(sep_min).detach(),
                "sep_mean_deg": torch.rad2deg(sep_mean).detach(),
                "anc_norm":     anc_norm.mean().detach(),
            }
        return loss, stats
