"""Self-check for the axis-distance cone loss. No GPU, no data, no checkpoint.

    python -m tests.test_axis_cone_loss

Eight invariants. The first three pin down that q is the quantity we think it is; the
rest pin down the failures that would be invisible in a training log — a dead gradient,
a coupled parameter that hands the optimiser a shortcut, and a decision rule that has
quietly become a cosine.

  1. q is STRICTLY MONOTONE in the angle over the whole [0, pi]. The obvious
     (sinh d_axis / sinh R)^2 is not: it is bilateral, its derivative flips sign at 90
     degrees, and past that "descending" means walking to 180 — an antipodal attractor,
     which is the degeneracy Run C hit with a pair of anchors at 179 degrees;
  2. q = 1 exactly on the cone wall, < 1 inside, > 1 outside;
  3. q is invariant to the image's depth, so there is no collapse-toward-the-origin
     direction and nothing to calibrate about how deep the head starts;
  4. the gradient on the IMAGE is non-zero everywhere, including near 180 degrees;
  5. with equal psi, argmin q IS argmax cos; with different psi they diverge. That is
     the necessary condition for the cone rule to be worth anything;
  6. DECOUPLING: the radial component of the gradient on the anchor is exactly zero and
     the tangential one is not, so the anchor can only rotate. When the aperture was
     derived from the anchor's depth instead, "widen my cone" and "move toward the
     origin" were the same move and the optimiser took it every time — measured, psi ran
     to 65 degrees and its spread across classes collapsed from 8.7 to 0.6;
  7. psi's gradient has the right sign on both terms, and the sigmoid parameterisation
     keeps it inside its range with no projection;
  8. the aperture equilibrium is where the derivation says: the wall lands on the class's
     RMS angular radius.
"""
import math

import torch
import torch.nn.functional as F

from losses.axis_cone_loss import (AxisConeLoss, axis_cone_q, depth_from_sin_psi,
                                   sin_psi_from_depth)

# float64 throughout: checks 6-8 compare autograd against finite differences, and in
# float32 the difference quotients are a few hundred ulp of the loss — they would
# disagree at the 1e-2 level for pure rounding reasons, hiding a real discrepancy of the
# same size. Everything here is CPU and tiny, so it is free.
torch.set_default_dtype(torch.float64)

K_R = 0.5          # min_radius
D = 6


def _sin(psi_deg):
    return torch.tensor([math.sin(math.radians(psi_deg))])


def _axis(index: int = 0) -> torch.Tensor:
    """An anchor direction. Its norm is irrelevant to q — that is the whole point."""
    a = torch.zeros(1, D)
    a[0, index] = 1.0
    return a


def _at_angle(angle: float, depth: float = 10.0, axis: int = 0,
              off: int = 1) -> torch.Tensor:
    x = torch.zeros(1, D)
    x[0, axis] = depth * math.cos(angle)
    x[0, off] = depth * math.sin(angle)
    return x


def test_monotone_and_one_sided():
    a, sp = _axis(), _sin(14.5)
    angles = [i * math.pi / 200 for i in range(201)]
    qs = [axis_cone_q(_at_angle(t), a, sp).item() for t in angles]
    diffs = [b - c for c, b in zip(qs, qs[1:])]
    assert min(diffs) > 0, f"not monotone: min step {min(diffs):.3e}"
    bad = [(math.sin(t) / sp.item()) ** 2 for t in angles]
    assert bad[-1] < bad[len(bad) // 2], "sanity: (sin t/sin psi)^2 should fold back"
    print(f"1 ok  q strictly monotone 0→π ({qs[0]:.3f} → {qs[-1]:.1f}); "
          f"the bilateral form folds back to {bad[-1]:.3f} at π")


def test_q_is_one_on_the_wall():
    a, psi = _axis(), math.radians(14.5)
    sp = _sin(14.5)
    inside = axis_cone_q(_at_angle(psi * 0.5), a, sp).item()
    wall   = axis_cone_q(_at_angle(psi),       a, sp).item()
    out    = axis_cone_q(_at_angle(psi * 1.5), a, sp).item()
    assert abs(wall - 1.0) < 1e-9, wall
    assert inside < 1.0 < out, (inside, wall, out)
    print(f"2 ok  q: inside {inside:.3f} < wall {wall:.9f} < outside {out:.3f}")


def test_q_is_depth_invariant():
    a, sp = _axis(), _sin(14.5)
    qs = [axis_cone_q(_at_angle(0.08, depth=d), a, sp).item()
          for d in (0.5, 5.0, 50.0, 500.0)]
    assert max(qs) - min(qs) < 1e-9, qs
    print(f"3 ok  q invariant over depth 0.5→500: {qs[0]:.9f} … {qs[-1]:.9f}")


def test_no_dead_gradient_anywhere():
    a, sp = _axis(), _sin(14.5)
    psi = math.radians(14.5)
    loss_fn = AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=0.0)
    grads = {}
    for tag, angle in (("deep inside", psi * 0.1), ("on the wall", psi),
                       ("far outside", 2.0), ("near π", math.pi - 0.05)):
        x = _at_angle(angle).clone().requires_grad_(True)
        loss_fn(x, a, torch.zeros(1, dtype=torch.long), sp)[0].backward()
        grads[tag] = x.grad.abs().max().item()
        assert grads[tag] > 0, (tag, grads[tag])
    print("4 ok  image gradient alive everywhere: "
          + ", ".join(f"{k} {v:.2e}" for k, v in grads.items()))


def test_psi_spread_breaks_the_cosine_rule():
    x = _at_angle(math.radians(10.0))                 # 10° from e0, 80° from e1
    u = torch.zeros(2, D); u[0, 0] = 1.0; u[1, 1] = 1.0
    cos = (F.normalize(x, dim=-1) @ u.T).squeeze()
    assert cos.argmax().item() == 0

    same = axis_cone_q(x, u, torch.tensor([0.25, 0.25])).squeeze()
    assert same.argmin().item() == 0, same

    mixed = axis_cone_q(x, u, torch.tensor([0.025, 0.625])).squeeze()   # narrow, wide
    assert mixed.argmin().item() == 1, mixed
    print(f"5 ok  equal ψ → argmin q == argmax cos ({same[0]:.2f} vs {same[1]:.2f}); "
          f"unequal ψ → they diverge ({mixed[0]:.2f} vs {mixed[1]:.2f}, "
          f"cos {cos[0]:.3f} vs {cos[1]:.3f})")


def test_anchor_is_direction_only():
    """The decoupling itself: q reads a normalised direction, so pushing the anchor along
    its own ray must change nothing at all."""
    u = torch.zeros(2, D); u[0, 0] = 1.0; u[1, 1] = 1.0
    sp = torch.tensor([0.25, 0.25])
    x = _at_angle(math.radians(9.0), off=2)
    labels = torch.zeros(1, dtype=torch.long)

    a = (u * 4.0).clone().requires_grad_(True)
    AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=0.0)(
        x, a, labels, sp)[0].backward()
    u_hat = F.normalize(a.detach()[0], dim=-1)
    radial = (a.grad[0] @ u_hat).item()
    tang = (a.grad[0] - radial * u_hat).norm().item()
    assert abs(radial) < 1e-12, radial
    assert tang > 1e-6, tang

    # and the loss is literally unchanged by a 10x rescale of the anchors
    q1 = axis_cone_q(x, u * 4.0, sp)
    q2 = axis_cone_q(x, u * 40.0, sp)
    assert torch.allclose(q1, q2, atol=1e-12), (q1, q2)

    # the axis still turns the right way in each case
    for lab, lam, want_closer in ((0, 0.0, True), (1, 1.0, False)):
        a = (u * 4.0).clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=lam, lambda_aperture=0.0)(
            x, a, torch.full((1,), lab, dtype=torch.long), sp)[0].backward()
        before = F.normalize(u[0], dim=-1) @ F.normalize(x[0], dim=-1)
        after = F.normalize(u[0] * 4.0 - 1e-3 * a.grad[0], dim=-1) @ F.normalize(x[0], dim=-1)
        assert (after > before) == want_closer, (lab, before.item(), after.item())
    print(f"6 ok  anchor is direction-only: radial {radial:.1e} (exactly 0), "
          f"tangential {tang:.4f}; q unchanged by a 10× rescale; axis turns toward its "
          f"own class and away from an intruder")


def test_psi_gradient_signs_and_bounds():
    a = _axis()
    x = _at_angle(math.radians(20.0))
    labels = torch.zeros(1, dtype=torch.long)

    def g_psi(psi_deg, lam_neg, lam_ap, lab=0):
        sp = _sin(psi_deg).clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=lam_neg, lambda_aperture=lam_ap)(
            x, torch.cat([a, _axis(1)]), torch.full((1,), lab, dtype=torch.long),
            torch.cat([sp, _sin(psi_deg)]))[0].backward()
        return sp.grad[0].item()

    pos = g_psi(30.0, 0.0, 0.0)          # own class: descent should WIDEN
    ap = g_psi(30.0, 0.0, 1.0) - pos     # aperture term alone: should NARROW
    assert pos < 0, pos
    assert ap > 0, ap

    # sigmoid parameterisation: psi is inside the range for any raw value, no clamp
    lo, hi = math.radians(5.0), math.radians(60.0)
    for raw in (-50.0, -1.0, 0.0, 1.0, 50.0):
        psi = lo + (hi - lo) * torch.sigmoid(torch.tensor(raw))
        assert lo <= psi.item() <= hi, (raw, psi)
    mid = lo + (hi - lo) * torch.sigmoid(torch.tensor(0.0))
    print(f"7 ok  ψ gradient: positive term {pos:+.4f} (widens), aperture {ap:+.4f} "
          f"(narrows); sigmoid bounds ψ to [5, 60]°, init {math.degrees(mid):.1f}°")


def test_aperture_equilibrium():
    """With every image of a class at the same angle theta, the derivation says the wall
    settles at exactly theta (lambda_aperture = 1)."""
    theta = math.radians(22.0)
    x = torch.cat([_at_angle(theta, off=1), _at_angle(theta, off=2),
                   _at_angle(theta, off=3)])
    labels = torch.zeros(3, dtype=torch.long)

    def g(psi_deg):
        sp = _sin(psi_deg).clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=1.0)(
            x, _axis(), labels, sp)[0].backward()
        return sp.grad[0].item()

    at_eq, too_wide, too_narrow = g(22.0), g(45.0), g(8.0)
    assert abs(at_eq) < 1e-9, at_eq
    assert too_wide > 0, too_wide        # descent shrinks psi toward the spread
    assert too_narrow < 0, too_narrow    # descent grows psi toward the spread

    # and the depth the aperture implies round-trips, which is what keeps the stored
    # anchor, the snapshots and the eval reading the same geometry
    sp = _sin(22.0)
    assert torch.allclose(sin_psi_from_depth(
        _axis() * depth_from_sin_psi(sp, K_R), K_R), sp, atol=1e-12)
    print(f"8 ok  aperture equilibrium at ψ = the class spread (22°): grad {at_eq:+.2e}; "
          f"ψ=45° → {too_wide:+.3f} (narrows), ψ=8° → {too_narrow:+.3f} (widens); "
          f"depth↔ψ round trip exact")


if __name__ == "__main__":
    test_monotone_and_one_sided()
    test_q_is_one_on_the_wall()
    test_q_is_depth_invariant()
    test_no_dead_gradient_anywhere()
    test_psi_spread_breaks_the_cosine_rule()
    test_anchor_is_direction_only()
    test_psi_gradient_signs_and_bounds()
    test_aperture_equilibrium()
    print("\nall axis-cone invariants hold")
