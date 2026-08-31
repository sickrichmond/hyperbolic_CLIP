"""Self-check for the axis-distance cone loss. No GPU, no data, no checkpoint.

    python -m tests.test_axis_cone_loss

Eleven invariants. The first three pin down that q is the quantity we think it is; the
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
  8. the aperture equilibrium is where the derivation says: the cone shrinks until the
     mass of the samples left outside it equals nu, so nu upper-bounds the fraction of a
     class's own samples that fall outside its own cone;
  9. the two crossed detaches hold — the pull term cannot move psi, and the aperture term
     cannot move the images. Without them, widening the cone is a cheaper way to lower
     the loss than rotating a 128-dimensional axis, and the optimiser takes it.
 10. overlapping cones pay exactly their squared angular violation, including the
     requested empty margin; disjoint cones pay zero.
 11. the inside margin is part of coverage, not the classification score.
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

    def g_psi(psi_deg, lam_neg, lam_ap, lab=0, nu=0.05):
        sp = _sin(psi_deg).clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=lam_neg, lambda_aperture=lam_ap, nu=nu)(
            x, torch.cat([a, _axis(1)]), torch.full((1,), lab, dtype=torch.long),
            torch.cat([sp, _sin(psi_deg)]))[0].backward()
        return sp.grad[0].item()

    # the pull term is detached from W, so it must not move psi AT ALL
    pull_only = g_psi(30.0, 0.0, 0.0)
    assert abs(pull_only) < 1e-12, pull_only
    # the image at 20 deg is inside a 30 deg cone, so no sample is outside: the hinge is
    # silent and log W alone is left, which narrows.
    ap = g_psi(30.0, 0.0, 1.0)
    assert ap > 0, ap

    # sigmoid parameterisation: psi is inside the range for any raw value, no clamp
    lo, hi = math.radians(5.0), math.radians(60.0)
    for raw in (-50.0, -1.0, 0.0, 1.0, 50.0):
        psi = lo + (hi - lo) * torch.sigmoid(torch.tensor(raw))
        assert lo <= psi.item() <= hi, (raw, psi)
    mid = lo + (hi - lo) * torch.sigmoid(torch.tensor(0.0))
    print(f"7 ok  pull cannot move ψ ({pull_only:.1e}, detached), aperture {ap:+.4f} "
          f"(narrows when nothing is outside); sigmoid bounds ψ to [5, 60]°, "
          f"init {math.degrees(mid):.1f}°")


def test_aperture_equilibrium_is_coverage():
    """nu bounds the fraction left outside. Build a class with a known spread of angles,
    find where the aperture gradient vanishes, and check the coverage there."""
    nu = 0.10
    n = 400
    # angles spread over [4, 44] degrees: a mix of tight and outlying samples
    angles = [math.radians(4.0 + 40.0 * i / (n - 1)) for i in range(n)]
    x = torch.cat([_at_angle(t, off=1 + (i % 4)) for i, t in enumerate(angles)])
    labels = torch.zeros(n, dtype=torch.long)

    def g(psi_deg):
        sp = _sin(psi_deg).clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=1.0, nu=nu)(
            x, _axis(), labels, sp)[0].backward()
        return sp.grad[0].item()

    lo, hi = 5.0, 59.0
    assert g(lo) < 0 and g(hi) > 0, (g(lo), g(hi))     # brackets a minimum
    for _ in range(60):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    psi_star = (lo + hi) / 2

    sp = _sin(psi_star)
    q = axis_cone_q(x, _axis(), sp).squeeze(1)
    outside = (q > 1).double().mean().item()
    viol_mass = (q * (q > 1)).mean().item()
    # The violator mass is a STEP function of psi: each sample crosses the wall at q = 1
    # and adds 1/n to it, so it jumps over nu rather than landing on it. One jump is the
    # honest tolerance here, not a round number.
    assert abs(viol_mass - nu) < 2.0 / n, (viol_mass, nu, 1.0 / n)
    assert outside <= nu + 1e-9, (outside, nu)

    # the depth the aperture implies round-trips, which is what keeps the stored anchor,
    # the snapshots and the eval reading the same geometry
    assert torch.allclose(sin_psi_from_depth(
        _axis() * depth_from_sin_psi(sp, K_R), K_R), sp, atol=1e-12)
    print(f"8 ok  coverage equilibrium at ψ={psi_star:.2f}°: viol mass {viol_mass:.4f} "
          f"= ν {nu} to within one sample ({1.0/n:.4f}), and {100*outside:.2f}% of the "
          f"class outside its own cone (≤ ν by construction); depth↔ψ round trip exact")


def test_detaches_hold():
    """Each parameter has one job: the pull must not reach psi, the aperture must not
    reach the images. This is what stops 'widen the cone' from substituting for 'rotate
    the axis' — the substitution that ran psi to 65 degrees when they were coupled."""
    x = _at_angle(math.radians(40.0)).clone().requires_grad_(True)   # outside a 30° cone
    sp = _sin(30.0).clone().requires_grad_(True)
    labels = torch.zeros(1, dtype=torch.long)

    AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=0.0, nu=0.05)(
        x, _axis(), labels, sp)[0].backward()
    assert x.grad.abs().max() > 0, "pull must move the image"
    assert sp.grad is None or abs(sp.grad.item()) < 1e-12, sp.grad

    x2 = _at_angle(math.radians(40.0)).clone().requires_grad_(True)
    sp2 = _sin(30.0).clone().requires_grad_(True)
    ap_only = AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=1.0, nu=0.05)
    (ap_only(x2, _axis(), labels, sp2)[0]
     - AxisConeLoss(min_radius=K_R, lambda_neg=0.0, lambda_aperture=0.0)(
         x2, _axis(), labels, sp2)[0]).backward()
    assert abs(sp2.grad.item()) > 0, "aperture must move psi"
    assert x2.grad is None or x2.grad.abs().max() < 1e-12, x2.grad.abs().max()
    print("9 ok  detaches hold: pull moves the image and not ψ, aperture moves ψ and not "
          "the image")


def test_overlap_penalty():
    psi_deg, margin_deg, angle_deg = 20.0, 5.0, 30.0
    anchors = torch.zeros(2, D)
    anchors[0, 0] = 1.0
    anchors[1, 0] = math.cos(math.radians(angle_deg))
    anchors[1, 1] = math.sin(math.radians(angle_deg))
    sin_psi = torch.full((2,), math.sin(math.radians(psi_deg)))
    labels = torch.zeros(1, dtype=torch.long)
    loss_fn = AxisConeLoss(
        min_radius=K_R, lambda_neg=0.0, lambda_aperture=0.0,
        lambda_sep=1.0, separation_margin=margin_deg,
    )

    _, stats = loss_fn(_axis(), anchors, labels, sin_psi)
    expected = math.radians(2 * psi_deg + margin_deg - angle_deg) ** 2
    assert abs(stats["loss_sep"].item() - expected) < 1e-12
    assert stats["sep_overlap"].item() == 1.0

    anchors[1].zero_()
    anchors[1, 1] = 1.0
    _, clear = loss_fn(_axis(), anchors, labels, sin_psi)
    assert clear["loss_sep"].item() == 0.0
    assert clear["sep_overlap"].item() == 0.0
    print("10 ok overlap penalty: 15° violation pays its squared angle; 90°-separated "
          "cones pay zero")


def test_inside_margin():
    x = _at_angle(math.radians(19.0))
    sin_psi = _sin(20.0)
    labels = torch.zeros(1, dtype=torch.long)
    _, plain = AxisConeLoss(
        min_radius=K_R, lambda_neg=0.0, lambda_aperture=1.0,
    )(x, _axis(), labels, sin_psi)
    _, padded = AxisConeLoss(
        min_radius=K_R, lambda_neg=0.0, lambda_aperture=1.0, inside_margin=2.0,
    )(x, _axis(), labels, sin_psi)
    assert plain["q_pos"].item() < 1.0
    assert plain["inside_img"].item() == 1.0
    assert padded["q_pos"].item() < 1.0, "classifier score must not change"
    assert padded["inside_img"].item() == 0.0, "coverage must include the margin"
    print("11 ok inside margin: a point 1° inside the wall still classifies inside, but "
          "fails a 2° padded coverage constraint")


if __name__ == "__main__":
    test_monotone_and_one_sided()
    test_q_is_one_on_the_wall()
    test_q_is_depth_invariant()
    test_no_dead_gradient_anywhere()
    test_psi_spread_breaks_the_cosine_rule()
    test_anchor_is_direction_only()
    test_psi_gradient_signs_and_bounds()
    test_aperture_equilibrium_is_coverage()
    test_detaches_hold()
    test_overlap_penalty()
    test_inside_margin()
    print("\nall axis-cone invariants hold")
