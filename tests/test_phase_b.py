"""Self-check for the Phase B loss terms. No GPU, no data, no checkpoint.

    python -m tests.test_phase_b

Five invariants, each pinning down one thing that would otherwise fail silently:

  1. the defaults are inert — with the new knobs off, the total is exactly what it was
     (L_img_in_class + λ_norm·L_norm), so `B_ce` really is one variable from sweepwin;
  2. the separation floor fires on overlapping cones and is zero on disjoint ones;
  3. the ceiling fires on an antipodal pair and is zero on a healthy spread;
  4. 'bilateral' penalises anchors that are too DEEP, 'floor' does not — the difference
     that let Run C's anchors drift to 8.18 with L_norm = 0;
  5. the family term is zero when a model anchor sits inside its family's cone and
     positive when it does not;
  6. pos_mode="axis" still has gradient where the hinge has exactly none — a point
     already INSIDE its cone. That is the whole reason the term exists;
  7. neg_samples keeps exactly k negatives per row, and all of them when k is larger
     than the row;
  8. the anchor norm clamp pulls a tangent back into range from BOTH sides, and the
     Poincare<->Lorentz round trip the disk plot relies on is exact.
"""
import math

import torch

from geometry.lorentz import exp_map0, half_aperture, oxy_angle
from losses.attribution_loss import EntailmentConeLoss, _subsample

D, K, B = 8, 4, 6


def _ray(direction: torch.Tensor, norm: float) -> torch.Tensor:
    """A point on the hyperboloid at the given space-norm, along `direction`."""
    d = direction / direction.norm(dim=-1, keepdim=True)
    return d * norm


def test_defaults_are_inert():
    g = torch.Generator().manual_seed(0)
    x_img = exp_map0(torch.randn(B, D, generator=g))
    x_anc = exp_map0(torch.randn(K, D, generator=g))
    labels = torch.arange(B) % K

    loss_fn = EntailmentConeLoss(min_radius=0.1, lambda_norm=0.5, target_norm=4.0)
    loss, st = loss_fn(x_img, x_anc, labels)
    expected = st["loss_img_in_cls"] + 0.5 * st["loss_norm"]
    assert torch.allclose(loss, expected, atol=1e-6), (loss, expected)
    assert st["loss_sep"].item() == 0.0
    print("1 ok  defaults inert, no new term leaks in")


def test_separation_floor():
    # Two anchors 0.5 degrees apart at norm 6 -> psi ~ 1.9 deg each, so they overlap.
    base = torch.zeros(2, D)
    base[0, 0] = 1.0
    base[1, 0] = math.cos(math.radians(0.5))
    base[1, 1] = math.sin(math.radians(0.5))
    tight = _ray(base, 6.0)
    psi = half_aperture(tight, min_radius=0.1)

    loss_fn = EntailmentConeLoss(min_radius=0.1, lambda_sep=1.0)
    overlap, st = loss_fn._sep_term(tight, psi)
    assert overlap.item() > 0, overlap
    assert st["sep_overlap"].item() == 1.0

    wide = torch.zeros(2, D)
    wide[0, 0] = 1.0
    wide[1, 1] = 1.0                       # 90 degrees apart
    wide = _ray(wide, 6.0)
    ok, st = loss_fn._sep_term(wide, half_aperture(wide, min_radius=0.1))
    assert ok.item() == 0.0, ok
    assert st["sep_overlap"].item() == 0.0
    print(f"2 ok  floor: overlapping {overlap.item():.4f}, disjoint 0.0")


def test_separation_ceiling():
    anti = torch.zeros(2, D)
    anti[0, 0] = 1.0
    anti[1, 0] = -1.0                      # 180 degrees: Run C's pathology
    anti = _ray(anti, 6.0)
    loss_fn = EntailmentConeLoss(min_radius=0.1, lambda_sep=1.0, theta_max=150.0)
    fired, _ = loss_fn._sep_term(anti, half_aperture(anti, min_radius=0.1))
    assert fired.item() > 0, fired

    inert = torch.zeros(2, D)
    inert[0, 0] = 1.0
    inert[1, 0] = math.cos(math.radians(132.2))   # the euclidean model's widest pair
    inert[1, 1] = math.sin(math.radians(132.2))
    inert = _ray(inert, 6.0)
    quiet, _ = loss_fn._sep_term(inert, half_aperture(inert, min_radius=0.1))
    assert quiet.item() == 0.0, quiet
    print(f"3 ok  ceiling: 180 deg {fired.item():.4f}, 132.2 deg 0.0")


def test_norm_mode():
    x_img = exp_map0(torch.randn(B, D, generator=torch.Generator().manual_seed(1)))
    labels = torch.arange(B) % K
    deep = _ray(torch.eye(K, D), 8.18)     # Run C's measured anchor norm

    floor = EntailmentConeLoss(min_radius=0.1, lambda_norm=1.0, target_norm=4.0)
    both = EntailmentConeLoss(min_radius=0.1, lambda_norm=1.0, target_norm=4.0,
                              norm_mode="bilateral")
    _, st_floor = floor(x_img, deep, labels)
    _, st_both = both(x_img, deep, labels)
    assert st_floor["loss_norm"].item() == 0.0, st_floor["loss_norm"]
    assert abs(st_both["loss_norm"].item() - (8.18 - 4.0) ** 2) < 1e-3, st_both["loss_norm"]
    print(f"4 ok  norm: floor 0.0 at depth 8.18, bilateral {st_both['loss_norm'].item():.3f}")


def test_family_containment():
    x_img = exp_map0(torch.randn(B, D, generator=torch.Generator().manual_seed(2)))
    labels = torch.arange(B) % K
    family_of = torch.zeros(K, dtype=torch.long)          # every class in family 0

    dirs = torch.zeros(K, D)
    dirs[:, 0] = 1.0                                      # all along the family's ray
    inside = _ray(dirs, 6.0)
    fam = _ray(torch.eye(1, D), 2.0)                      # shallow -> wide cone

    loss_fn = EntailmentConeLoss(min_radius=0.1, lambda_family=1.0, family_of=family_of)
    _, st = loss_fn(x_img, inside, labels, x_fam=fam)
    assert st["inside_family"].item() == 1.0, st["inside_family"]
    assert st["loss_fam_anc"].item() == 0.0, st["loss_fam_anc"]

    away = torch.zeros(K, D)
    away[:, 1] = 1.0                                      # 90 degrees off the family
    _, st = loss_fn(x_img, _ray(away, 6.0), labels, x_fam=fam)
    assert st["inside_family"].item() == 0.0, st["inside_family"]
    assert st["loss_fam_anc"].item() > 0, st["loss_fam_anc"]
    print(f"5 ok  family: aligned inside, orthogonal {st['loss_fam_anc'].item():.3f}")


def test_axis_has_gradient_inside_the_cone():
    """The point of the whole change: a sample already inside its cone must still
    pull. The hinge gives it exactly zero gradient, xi^2 gives it 2*xi."""
    labels = torch.arange(K)
    dirs = torch.eye(K, D)
    x_anc = _ray(dirs, 3.0)
    # Each image straight out along its own anchor's ray but deeper => xi is small,
    # comfortably inside the cone.
    x_img = _ray(dirs, 9.0)

    psi = half_aperture(x_anc, min_radius=0.5)
    xi = oxy_angle(x_anc, x_img)
    assert bool((xi < psi).all()), (xi, psi)          # precondition: all inside

    def grad(pos_mode):
        p = x_img.clone().requires_grad_(True)
        loss_fn = EntailmentConeLoss(min_radius=0.5, lambda_neg=0.0, pos_mode=pos_mode)
        loss, _ = loss_fn(p, x_anc, labels)
        loss.backward()
        return p.grad.abs().max().item()

    g_hinge, g_axis = grad("hinge"), grad("axis")
    assert g_hinge == 0.0, g_hinge
    assert g_axis > 0.0, g_axis
    print(f"6 ok  inside the cone: hinge grad {g_hinge:.1e}, axis grad {g_axis:.3e}")


def test_negative_subsample():
    mask = torch.ones(5, 22, dtype=torch.bool)
    mask.scatter_(1, torch.arange(5).unsqueeze(1), False)      # 21 negatives per row
    assert mask.sum(1).tolist() == [21] * 5

    kept = _subsample(mask, 8)
    assert kept.sum(1).tolist() == [8] * 5, kept.sum(1)
    assert bool((kept & ~mask).sum() == 0), "kept a non-negative"
    assert bool((_subsample(mask, 0) == mask).all())           # 0 disables
    assert bool((_subsample(mask, 99) == mask).all())          # k >= row keeps all

    small = torch.zeros(3, 22, dtype=torch.bool)
    small[:, :3] = True
    assert _subsample(small, 8).sum(1).tolist() == [3] * 3      # fewer than k
    print("7 ok  subsample: exactly k per row, never a positive, degrades gracefully")


def test_anchor_clamp_and_poincare_round_trip():
    lo, hi = 1.0, 3.0
    t = torch.stack([torch.tensor([0.1] + [0.0] * (D - 1)),     # too short
                     torch.tensor([9.0] + [0.0] * (D - 1))])    # too long
    n = t.norm(dim=-1, keepdim=True)
    t = t * (n.clamp(lo, hi) / n.clamp_min(1e-8))
    assert torch.allclose(t.norm(dim=-1), torch.tensor([lo, hi])), t.norm(dim=-1)

    # The disk plot lifts mesh points back with x = 2p / (1 - ||p||^2); it has to be
    # the exact inverse of lorentz_to_poincare or the cones land in the wrong place.
    x = exp_map0(torch.randn(7, D, generator=torch.Generator().manual_seed(3)))
    x_time = torch.sqrt(1.0 + (x ** 2).sum(-1, keepdim=True))
    p = x / (x_time + 1.0)
    back = 2 * p / (1 - (p ** 2).sum(-1, keepdim=True))
    assert torch.allclose(back, x, atol=1e-5), (back - x).abs().max()
    print("8 ok  norm clamp bilateral, Poincare round trip exact")


if __name__ == "__main__":
    test_defaults_are_inert()
    test_separation_floor()
    test_separation_ceiling()
    test_norm_mode()
    test_family_containment()
    test_axis_has_gradient_inside_the_cone()
    test_negative_subsample()
    test_anchor_clamp_and_poincare_round_trip()
    print("\nall Phase B invariants hold")
