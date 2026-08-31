"""
Hierarchical entailment-cone loss for attribution (HySAC-style).

Hierarchy:
    class anchor (e.g. "A real image")               — broadest cone
        ⊃ augmented caption ("Real image of ...")    — narrower cone, content-specific
            ⊃ image embedding                         — leaf

Loss terms (all use the same cone-violation primitive):

  L_img_in_class:  image i must lie inside the cone of its class anchor y_i,
                   and outside the cones of all other class anchors.
                   (this is the term used at inference)

  L_cap_in_class:  caption i must lie inside the cone of its class anchor y_i,
                   and outside the cones of all other class anchors.

  L_img_in_cap:    image i must lie inside the cone of its OWN augmented caption,
                   and outside the cones of all other captions in the batch
                   (which differ in content and/or class).

For each term:
  L_pos = max(0, ξ_pos - ψ_pos)          pos_mode="hinge"  (default)
  L_pos = ξ_pos²                          pos_mode="axis"
  L_neg = max(0, ψ_neg + margin - ξ_neg)
where ξ = oxy_angle(apex, point) and ψ = half_aperture(apex).

pos_mode="axis" is an MSE from the cone AXIS rather than from its boundary. The
hinge goes to zero gradient the moment the point is inside the cone, which is the
measured reason the projection head is free to collapse the class geometry the
LoRA built (centroid ARI 0.253 -> -0.007); ξ² keeps pulling with gradient 2ξ
everywhere and vanishes only on the axis itself. ψ does not appear in it, so the
loss cannot be lowered by widening the cone — the aperture survives only in L_neg
and L_sep, where it is a constraint rather than a target.

neg_samples > 0 keeps a random subset of that many negatives per row instead of
all K-1. Note this does NOT save compute: xi is computed against every anchor
anyway (cone_acc and the CE term read the full row). It changes the OBJECTIVE —
dropout on the negative term.

Optional ranking term (λ_ce > 0):
  L_ce = CE(softmax(-ξ_img_anc / τ), y),  τ = softplus(param), LEARNED.
Why it is not redundant with the hinges: L_pos saturates the moment the image is
inside its own cone, and L_neg is averaged over K-1 classes so a single wrong class
carries λ_neg/(K-1). Neither optimises the ORDER of the ξ across classes, which is
exactly what inference (argmin ξ) reads. Inference stays untouched — argmin is
scale-invariant, so τ never leaves training.

Phase B adds three things, all aimed at one measured failure: the projection head
zeroes the class geometry the LoRA built (centroid ARI 0.253 -> -0.007 against the
generator taxonomy), and it does so because a SATURATING hinge gives it permission —
the same head trained with a plain CE keeps 0.119 of it.

  lambda_hinge      scale on L_img_in_class, so the hinge can be turned OFF and
                    replaced by the CE ranking term rather than mixed with it. Pure CE
                    produces well-conditioned anchors; CE grafted onto a hinge does not
                    (Run C: pairs at 8.8 deg AND pairs at 179 deg).
  norm_mode         'bilateral' makes the anchor norm a TARGET instead of a floor. The
                    floor is satisfied at any norm above it, which is how Run C drifted
                    to 8.18 with L_norm=0 while sweepwin sat at 4.11.
  lambda_sep        anchors must be at least psi_c + psi_c' apart (cones disjoint) and
                    at most theta_max apart (no antipodal waste). Imposed on the
                    PROJECTED anchors, i.e. downstream of the head, which is where the
                    separation is destroyed.
  lambda_family     family anchors, shallower and therefore wider, containing the model
                    anchors. This is the only construction in which psi varies across
                    anchors -- and with equal psi, argmin xi IS argmax cos, which is why
                    the cone rule has never differed from a cosine.

lambda_axis > 0 adds the AXIS-RAY regulariser, the always-on companion to the hinge:

  L_axis = mean_i d_ray(x_i, axis of a_{y_i})

d_ray is the geodesic distance to the cone's axis RAY — the geodesic from the origin
through the apex, restricted to the side the cone opens toward (geometry/lorentz.py).
It answers the standing objection to the hinge, which is that its gradient is exactly
zero the moment a point is inside its cone: d_ray keeps pulling all the way to the axis
and vanishes only ON it. Unlike pos_mode="axis" (L_pos = ξ²) it is a genuine hyperbolic
distance rather than an angle, and unlike the origin-angle score in axis_cone_loss it
READS THE RADIUS — so it cannot report a point as on-axis when that point is nowhere
near the cone.

Two properties it is chosen for, both asserted in tests/test_axis_ray_dist.py:
  - it is strictly monotone in the angle over all of [0, pi]. Measuring to the full
    geodesic instead is bilateral and makes the antipode an attractor;
  - its apex branch moves the anchor RADIALLY, which is the only term here that does.
    With psi coupled to depth (psi = asin(2K/‖a‖)) that is what lets each class find its
    own aperture, instead of L_norm pinning all K anchors to one norm and hence to one
    psi — and with equal psi, argmin xi IS argmax cos algebraically.
  The apex branch also pulls an image that is SHALLOWER than its anchor back outward,
  which is the one configuration where oxy_angle saturates at pi and the hinge has no
  gradient at all.

Total:
  L = lambda_hinge * L_img_in_class
      + λ_cap_in_class * L_cap_in_class
      + λ_img_in_cap   * L_img_in_cap
      + λ_norm         * L_norm   (anchor norm regulariser)
      + λ_axis         * L_axis   (distance to the cone axis ray)
      + λ_ce           * L_ce     (ranking / calibration term)
      + λ_sep          * L_sep    (cone disjointness, floor and ceiling)
      + λ_family       * L_family (model anchor in family cone + image in family cone)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.lorentz import axis_ray_dist, half_aperture, oxy_angle


def _pairwise_xi(apex: torch.Tensor, point: torch.Tensor, curv: float) -> torch.Tensor:
    """Pairwise oxy_angle: result[a, p] = oxy_angle(apex[a], point[p]).
    apex (A, D), point (P, D) → (A, P)."""
    A, D = apex.shape
    P, _ = point.shape
    apex_t  = apex.unsqueeze(1).expand(A, P, D).reshape(A * P, D)
    point_t = point.unsqueeze(0).expand(A, P, D).reshape(A * P, D)
    return oxy_angle(apex_t, point_t, curv=curv).reshape(A, P)


def _subsample(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Keep k random True entries per row (all of them if the row has fewer)."""
    if k <= 0 or k >= mask.shape[1]:
        return mask
    noise = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1.0)
    keep = noise.topk(k, dim=1).indices
    return torch.zeros_like(mask).scatter_(1, keep, True) & mask


class EntailmentConeLoss(nn.Module):
    def __init__(
        self,
        curv: float = 1.0,
        min_radius: float = 0.1,
        margin: float = 0.1,
        lambda_neg: float = 1.0,
        lambda_cap_in_class: float = 0.0,
        lambda_img_in_cap: float = 0.0,
        lambda_norm: float = 0.0,
        target_norm: float = 0.0,
        lambda_axis: float = 0.0,
        lambda_ce: float = 0.0,
        ce_tau_init: float = 1.0,
        lambda_hinge: float = 1.0,
        norm_mode: str = "floor",
        target_norm_family: float = 0.0,
        lambda_sep: float = 0.0,
        separation_margin: float = 0.0,
        theta_max: float = 150.0,
        lambda_family: float = 0.0,
        family_of: torch.Tensor | None = None,
        pos_mode: str = "hinge",
        neg_samples: int = 0,
    ):
        """
        lambda_cap_in_class > 0 and lambda_img_in_cap > 0 enable the hierarchical
        terms. They require x_cap to be passed to forward(). With both at 0
        (default) the loss reduces to image-in-class-anchor only.

        lambda_norm, target_norm: anchor-norm regulariser.
          L_norm = mean_c max(0, target_norm - ‖t_c‖)²

        lambda_ce > 0 adds the CE ranking term with a LEARNED temperature. The
        parameter is created only in that case, so at lambda_ce=0 the module still
        has no parameters at all and returns the same value it did before.

        pos_mode: "hinge" (default, unchanged) or "axis" (L_pos = ξ²).
        neg_samples: 0 (default, all K-1 negatives) or k random negatives per row.
        Both defaults are inert — every earlier run reproduces bit for bit.
        """
        super().__init__()
        self.curv = curv
        self.min_radius = min_radius
        self.margin = margin
        self.lambda_neg = lambda_neg
        self.lambda_cap_in_class = lambda_cap_in_class
        self.lambda_img_in_cap = lambda_img_in_cap
        self.lambda_norm = lambda_norm
        self.target_norm = target_norm
        self.lambda_axis = lambda_axis
        self.lambda_ce = lambda_ce
        self.lambda_hinge = lambda_hinge
        self.norm_mode = norm_mode
        self.target_norm_family = target_norm_family
        self.lambda_sep = lambda_sep
        self.separation_margin = math.radians(separation_margin)
        self.theta_max = math.radians(theta_max)
        self.lambda_family = lambda_family
        self.pos_mode = pos_mode
        self.neg_samples = neg_samples
        self.register_buffer("family_of", family_of)
        if lambda_ce > 0:
            # τ = softplus(raw) keeps the temperature positive without a clamp.
            raw = torch.tensor(float(ce_tau_init)).expm1().clamp(min=1e-6).log()
            self.ce_tau_raw = nn.Parameter(raw)
        if lambda_family > 0:
            # Its own temperature: family anchors sit closer to the origin, so their
            # exterior angles live on a different scale than the model anchors'.
            raw = torch.tensor(float(ce_tau_init)).expm1().clamp(min=1e-6).log()
            self.fam_tau_raw = nn.Parameter(raw)

    def _cone_term(
        self,
        xi_pos: torch.Tensor,      # (B,)        positive exterior angles
        psi_pos: torch.Tensor,     # (B,)        cone aperture at the positive apex
        xi_neg: torch.Tensor,      # (B, M)      exterior angles to all candidate apices
        psi_neg_b: torch.Tensor,   # (B, M)      cone apertures at candidate apices
        neg_mask: torch.Tensor,    # (B, M) bool only true where the apex is a negative
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (L_pos, L_neg) — both scalars."""
        if self.pos_mode == "axis":
            # MSE from the axis: gradient 2ξ everywhere, zero only ON the axis.
            # psi is absent, so the cone cannot be widened to lower the loss.
            L_pos = xi_pos.pow(2).mean()
        else:
            L_pos = torch.clamp(xi_pos - psi_pos, min=0.0).mean()
        neg_mask = _subsample(neg_mask, self.neg_samples)
        if neg_mask.any():
            L_neg = torch.clamp(psi_neg_b + self.margin - xi_neg, min=0.0)[neg_mask].mean()
        else:
            L_neg = torch.tensor(0.0, device=xi_pos.device)
        return L_pos, L_neg

    def _sep_term(self, x_anc: torch.Tensor, psi_anc: torch.Tensor):
        """Cones disjoint, and no antipodal waste. Returns (loss, stats).

        floor:   ∠(a_c, a_c') ≥ ψ_c + ψ_c'. This is the disjointness criterion the
                 trainer already checks at init (train_attribution.py:517) — and every
                 text-anchor run measured violates it by 12-17x, so no containment
                 statement the cones make is worth anything today.
        ceiling: ∠ ≤ theta_max. Run C reached CE-like spread but put one pair at 179°,
                 which wastes the sphere with 22 anchors in 128 dimensions. The
                 euclidean model — the one healthy configuration measured — has mean
                 92.6° (the simplex ideal arccos(-1/21) is 92.7°) and its widest pair at
                 132.2°, so a 150° ceiling is inert on a healthy set and fires only on
                 that pathology.
        """
        K = x_anc.shape[0]
        d = F.normalize(x_anc, dim=-1)
        cos = (d @ d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        iu = torch.triu_indices(K, K, offset=1, device=x_anc.device)
        ang = torch.arccos(cos[iu[0], iu[1]])
        need = psi_anc[iu[0]] + psi_anc[iu[1]] + self.separation_margin
        floor = torch.clamp(need - ang, min=0.0).pow(2).mean()
        ceiling = torch.clamp(ang - self.theta_max, min=0.0).pow(2).mean()
        with torch.no_grad():
            stats = {
                "sep_min_deg": torch.rad2deg(ang.min()).detach(),
                "sep_max_deg": torch.rad2deg(ang.max()).detach(),
                "sep_overlap": (ang < need).float().mean().detach(),
            }
        return floor + ceiling, stats

    def forward(
        self,
        x_img: torch.Tensor,                     # (B, D)
        x_anc: torch.Tensor,                     # (K, D)
        labels: torch.Tensor,                    # (B,) int in [0, K)
        x_cap: torch.Tensor | None = None,       # (B, D) augmented captions, optional
        x_fam: torch.Tensor | None = None,       # (F, D) family anchors, optional
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, _ = x_img.shape
        K, _ = x_anc.shape
        device = x_img.device

        psi_anc = half_aperture(x_anc, curv=self.curv, min_radius=self.min_radius)   # (K,)
        psi_anc_b = psi_anc.unsqueeze(0).expand(B, K)                                # (B, K)
        pos_idx   = labels.unsqueeze(1)                                              # (B, 1)
        psi_anc_pos = psi_anc_b.gather(1, pos_idx).squeeze(1)                        # (B,)

        neg_mask_anc = torch.ones(B, K, device=device, dtype=torch.bool)
        neg_mask_anc.scatter_(1, pos_idx, False)

        # ───── 1) L_img_in_class ─────────────────────────────────────────────
        # xi_ia[i, c] = oxy_angle(anchor_c, img_i)
        xi_ia = _pairwise_xi(x_anc, x_img, curv=self.curv).T                          # (B, K)
        xi_ia_pos = xi_ia.gather(1, pos_idx).squeeze(1)                               # (B,)
        L_imgcls_pos, L_imgcls_neg = self._cone_term(
            xi_ia_pos, psi_anc_pos, xi_ia, psi_anc_b, neg_mask_anc
        )
        L_img_in_class = L_imgcls_pos + self.lambda_neg * L_imgcls_neg

        with torch.no_grad():
            inside_img = (xi_ia_pos < psi_anc_pos).float().mean()
            cone_acc   = (xi_ia.argmin(dim=1) == labels).float().mean()
            # Fraction of the xi matrix sitting at oxy_angle's acos clamp. xi = pi means
            # the image is SHALLOWER than the anchor — the far side of the cone — and the
            # clamp makes the gradient there exactly zero, not merely small. Anything
            # above a few percent and the run is not learning, whatever the lr says.
            xi_sat = (xi_ia > math.pi - 5e-3).float().mean()

        # ───── 2 & 3) hierarchical caption-based terms (optional) ────────────
        L_cap_in_class = torch.tensor(0.0, device=device)
        L_img_in_cap   = torch.tensor(0.0, device=device)
        stats_extra = {}

        use_caps = (
            x_cap is not None
            and (self.lambda_cap_in_class > 0 or self.lambda_img_in_cap > 0)
        )
        if use_caps:
            # 2) L_cap_in_class — caption inside its class anchor's cone
            xi_ca = _pairwise_xi(x_anc, x_cap, curv=self.curv).T                      # (B, K)
            xi_ca_pos = xi_ca.gather(1, pos_idx).squeeze(1)
            L_capcls_pos, L_capcls_neg = self._cone_term(
                xi_ca_pos, psi_anc_pos, xi_ca, psi_anc_b, neg_mask_anc
            )
            L_cap_in_class = L_capcls_pos + self.lambda_neg * L_capcls_neg

            # 3) L_img_in_cap — image inside its OWN caption's cone; other batch
            #    captions act as negatives.
            psi_cap = half_aperture(x_cap, curv=self.curv, min_radius=self.min_radius)  # (B,)
            # xi_ic[i, j] = oxy_angle(cap_j, img_i)
            xi_ic = _pairwise_xi(x_cap, x_img, curv=self.curv).T                       # (B, B)
            xi_ic_pos = xi_ic.diagonal()                                               # (B,)
            psi_cap_b = psi_cap.unsqueeze(0).expand(B, B)                              # (B, B)
            neg_mask_ic = ~torch.eye(B, dtype=torch.bool, device=device)
            L_imgcap_pos, L_imgcap_neg = self._cone_term(
                xi_ic_pos, psi_cap, xi_ic, psi_cap_b, neg_mask_ic
            )
            L_img_in_cap = L_imgcap_pos + self.lambda_neg * L_imgcap_neg

            with torch.no_grad():
                inside_cap     = (xi_ca_pos < psi_anc_pos).float().mean()
                inside_img_cap = (xi_ic_pos < psi_cap).float().mean()
                stats_extra = {
                    "inside_cap":      inside_cap.detach(),
                    "inside_img_cap":  inside_img_cap.detach(),
                    "mean_psi_cap":    psi_cap.mean().detach(),
                    "mean_xi_cap_anc": xi_ca_pos.mean().detach(),
                    "mean_xi_img_cap": xi_ic_pos.mean().detach(),
                    "mean_cap_norm":   x_cap.norm(dim=-1).mean().detach(),
                }

        # ───── 4) Anchor-norm regulariser ────────────────────────────────────
        anc_norms = x_anc.norm(dim=-1)
        if self.lambda_norm > 0 and self.target_norm > 0:
            if self.norm_mode == "bilateral":
                L_norm = (anc_norms - self.target_norm).pow(2).mean()
                if x_fam is not None and self.target_norm_family > 0:
                    L_norm = L_norm + (
                        x_fam.norm(dim=-1) - self.target_norm_family
                    ).pow(2).mean()
            else:
                L_norm = torch.clamp(self.target_norm - anc_norms, min=0.0).pow(2).mean()
        else:
            L_norm = torch.tensor(0.0, device=device)

        # ───── 4b) Axis-ray regulariser ──────────────────────────────────────
        # The hinge above is exactly flat inside the cone; this is not. d_ray vanishes
        # only ON the axis and reads the RADIUS, so unlike an origin-angle score it
        # cannot call a point on-axis while that point sits nowhere near the cone.
        L_axis = torch.tensor(0.0, device=device)
        if self.lambda_axis > 0:
            a_pos = x_anc[labels]                                            # (B, D)
            L_axis = axis_ray_dist(x_img, a_pos, curv=self.curv).mean()
            with torch.no_grad():
                # The direct monitor for the failure this term exists to prevent: an
                # entailment cone holds the points DEEPER than its apex, so any image
                # shallower than its own anchor is outside every cone by construction and
                # sits in oxy_angle's acos clamp, where the hinge gradient is zero. A run
                # with this above a few percent is not training, whatever `inside` says.
                frac_shallow = (x_img.norm(dim=-1) <= a_pos.norm(dim=-1)).float().mean()
                stats_extra.update({
                    "loss_axis":    L_axis.detach(),
                    "frac_shallow": frac_shallow.detach(),
                })

        # ───── 6) Anchor separation ──────────────────────────────────────────
        _d = F.normalize(x_anc, dim=-1)
        _iu = torch.triu_indices(x_anc.shape[0], x_anc.shape[0], offset=1, device=device)
        sep_ang = torch.arccos((_d @ _d.T).clamp(-1.0 + 1e-6, 1.0 - 1e-6)[_iu[0], _iu[1]])

        L_sep = torch.tensor(0.0, device=device)
        if self.lambda_sep > 0:
            L_sep, sep_stats = self._sep_term(x_anc, psi_anc)
            stats_extra.update(sep_stats)

        # ───── 7) Hierarchy: family cones containing the model cones ─────────
        L_family = torch.tensor(0.0, device=device)
        if self.lambda_family > 0 and x_fam is not None:
            psi_fam = half_aperture(x_fam, curv=self.curv, min_radius=self.min_radius)
            fam_of = self.family_of
            fam_labels = fam_of[labels]
            # (a) each model anchor inside its family's cone. A hinge is the right
            #     shape HERE: it is a containment constraint on 22 anchors that should
            #     be met and then stop pulling.
            xi_mf = oxy_angle(x_fam[fam_of], x_anc, curv=self.curv)              # (K,)
            L_mf = torch.clamp(xi_mf - psi_fam[fam_of], min=0.0).mean()
            # (b) each image inside its family's cone. CE, NOT a hinge: the whole point
            #     of turning the hinge off is that saturation lets the projection head
            #     collapse the class geometry (centroid ARI 0.253 -> -0.007), and a
            #     hinge here would reintroduce exactly that on the level that matters.
            xi_if = _pairwise_xi(x_fam, x_img, curv=self.curv).T                 # (B, F)
            L_if = F.cross_entropy(-xi_if / F.softplus(self.fam_tau_raw), fam_labels)
            L_family = L_mf + L_if
            with torch.no_grad():
                stats_extra.update({
                    "loss_fam_anc":  L_mf.detach(),
                    "loss_fam_img":  L_if.detach(),
                    "inside_family": (xi_mf < psi_fam[fam_of]).float().mean().detach(),
                    "family_acc":    (xi_if.argmin(1) == fam_labels).float().mean().detach(),
                    "mean_psi_fam":  psi_fam.mean().detach(),
                    "fam_tau":       F.softplus(self.fam_tau_raw).detach(),
                })

        loss = (
            self.lambda_hinge          * L_img_in_class
            + self.lambda_cap_in_class * L_cap_in_class
            + self.lambda_img_in_cap   * L_img_in_cap
            + self.lambda_norm         * L_norm
            + self.lambda_axis         * L_axis
            + self.lambda_sep          * L_sep
            + self.lambda_family       * L_family
        )

        # ───── 5) CE ranking term (optional) ─────────────────────────────────
        # -xi_ia is exactly the logit matrix the eval builds (test_hypclip.py:155),
        # so this trains the quantity inference actually ranks.
        if self.lambda_ce > 0:
            tau = F.softplus(self.ce_tau_raw)
            L_ce = F.cross_entropy(-xi_ia / tau, labels)
            loss = loss + self.lambda_ce * L_ce
            stats_extra["loss_ce"] = L_ce.detach()
            stats_extra["ce_tau"]  = tau.detach()

        stats = {
            "loss_img_in_cls": L_img_in_class.detach(),
            # Split out: with pos_mode="axis" the two halves live on different
            # scales and the sum alone is unreadable.
            "xi_sat":          xi_sat.detach(),
            "loss_pos":        L_imgcls_pos.detach(),
            "loss_neg":        L_imgcls_neg.detach(),
            "loss_cap_in_cls": L_cap_in_class.detach(),
            "loss_img_in_cap": L_img_in_cap.detach(),
            "loss_norm":       L_norm.detach(),
            "loss_sep":        L_sep.detach(),
            "inside_img":      inside_img.detach(),
            "cone_acc":        cone_acc.detach(),
            "mean_psi_anc":    psi_anc.mean().detach(),
            # psi SPREAD, not just its mean: with every leaf at the same depth psi is
            # uniform and argmin xi IS argmax cos algebraically. A spread is the
            # necessary condition for the cone rule to differ from a cosine at all.
            "psi_min_deg":     torch.rad2deg(psi_anc.min()).detach(),
            "psi_max_deg":     torch.rad2deg(psi_anc.max()).detach(),
            # Pairwise anchor separation, logged ALWAYS and not only under lambda_sep:
            # this is the collapse number (78.7 deg at random init -> 41.2 within one
            # epoch in the previous run) and the 2-D snapshots cannot show it, since 22
            # near-orthogonal directions in 128-d project to one blob whatever they do.
            "sep_min_deg":     torch.rad2deg(sep_ang.min()).detach(),
            "sep_mean_deg":    torch.rad2deg(sep_ang.mean()).detach(),
            "mean_xi_img_anc": xi_ia_pos.mean().detach(),
            "mean_anc_norm":   anc_norms.mean().detach(),
            # a pair overlaps when its axes are closer than the sum of its apertures
            "sep_overlap":     (sep_ang < psi_anc[_iu[0]] + psi_anc[_iu[1]]
                                 + self.separation_margin
                                ).float().mean().detach(),
            **stats_extra,
        }
        return loss, stats


def predict_class(x_img: torch.Tensor, x_anc: torch.Tensor, curv: float = 1.0) -> torch.Tensor:
    """Image-only inference: pick the anchor with smallest exterior angle."""
    xi = _pairwise_xi(x_anc, x_img, curv=curv).T   # (B, K)
    return xi.argmin(dim=1)
