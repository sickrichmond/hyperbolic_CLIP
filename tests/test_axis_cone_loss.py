"""Self-check for the axis-distance cone loss. No GPU, no data, no checkpoint.

    python -m tests.test_axis_cone_loss

Seven invariants. The first three pin down that q is the quantity we think it is; the
rest pin down the failures that would be invisible in a training log — a dead gradient,
a dead anchor channel, and a decision rule that has quietly become a cosine.

  1. q is STRICTLY MONOTONE in the angle over the whole [0, pi]. The obvious
     (sinh d_axis / sinh R)^2 is not: it is bilateral, its derivative flips sign at 90
     degrees, and past that "descending" means walking to 180 — an antipodal attractor,
     which is the degeneracy Run C hit with a pair of anchors at 179 degrees;
  2. q = 1 exactly on the cone wall, < 1 inside, > 1 outside;
  3. q is invariant to the image's depth, so there is no collapse-toward-the-origin
     direction and nothing to calibrate about how deep the head starts;
  4. the gradient on the IMAGE is non-zero everywhere, including near 180 degrees. Both
     previous formulations died on a clamp; this one has no transcendental function;
  5. with equal psi, argmin q IS argmax cos; with different psi they diverge. That is
     the necessary condition for the cone rule to be worth anything;
  6. BOTH anchor gradient channels agree with a finite difference (a stray detach in the
     normalisation would silently freeze the radius, psi would go uniform, and we would
     be back to a cosine without any sign of it), and both have the right sign;
  7. the norm clamp pulls the anchor radius back into range from both sides.
"""
import math

import torch
import torch.nn.functional as F

from losses.axis_cone_loss import AxisConeLoss, axis_cone_q

K_R = 0.5          # min_radius
D = 6


def _axis(index: int = 0, norm: float = 4.0) -> torch.Tensor:
    a = torch.zeros(1, D)
    a[0, index] = norm
    return a


def _at_angle(angle: float, depth: float = 10.0, axis: int = 0,
              off: int = 1) -> torch.Tensor:
    """A point at hyperboloid norm `depth`, `angle` radians off the axis e_axis."""
    x = torch.zeros(1, D)
    x[0, axis] = depth * math.cos(angle)
    x[0, off] = depth * math.sin(angle)
    return x


def test_monotone_and_one_sided():
    a = _axis()
    angles = [i * math.pi / 200 for i in range(201)]
    qs = [axis_cone_q(_at_angle(t), a, K_R).item() for t in angles]
    diffs = [b - c for c, b in zip(qs, qs[1:])]
    assert min(diffs) > 0, f"not monotone: min step {min(diffs):.3e}"
    assert qs[-1] > qs[0], (qs[0], qs[-1])
    # and the bilateral alternative is NOT monotone — the thing this replaces
    sin_psi = 2 * K_R / 4.0
    bad = [(math.sin(t) / sin_psi) ** 2 for t in angles]
    assert bad[-1] < bad[len(bad) // 2], "sanity: (sin t/sin psi)^2 should fold back"
    print(f"1 ok  q strictly monotone 0→π ({qs[0]:.3f} → {qs[-1]:.1f}); "
          f"the bilateral form folds back to {bad[-1]:.3f} at π")


def test_q_is_one_on_the_wall():
    a = _axis()
    psi = math.asin(2 * K_R / 4.0)
    inside = axis_cone_q(_at_angle(psi * 0.5), a, K_R).item()
    wall   = axis_cone_q(_at_angle(psi),       a, K_R).item()
    out    = axis_cone_q(_at_angle(psi * 1.5), a, K_R).item()
    assert abs(wall - 1.0) < 1e-5, wall
    assert inside < 1.0 < out, (inside, wall, out)
    print(f"2 ok  q: inside {inside:.3f} < wall {wall:.6f} < outside {out:.3f}  "
          f"(ψ={math.degrees(psi):.1f}°)")


def test_q_is_depth_invariant():
    a = _axis()
    qs = [axis_cone_q(_at_angle(0.08, depth=d), a, K_R).item()
          for d in (0.5, 5.0, 50.0, 500.0)]
    assert max(qs) - min(qs) < 1e-5, qs
    print(f"3 ok  q invariant over depth 0.5→500: {qs[0]:.6f} … {qs[-1]:.6f}")


def test_no_dead_gradient_anywhere():
    a = _axis()
    psi = math.asin(2 * K_R / 4.0)
    loss_fn = AxisConeLoss(min_radius=K_R, lambda_neg=0.0)
    grads = {}
    for tag, angle in (("deep inside", psi * 0.1), ("on the wall", psi),
                       ("far outside", 2.0), ("near π", math.pi - 0.05)):
        x = _at_angle(angle).clone().requires_grad_(True)
        loss, _ = loss_fn(x, a, torch.zeros(1, dtype=torch.long))
        loss.backward()
        grads[tag] = x.grad.abs().max().item()
        assert grads[tag] > 0, (tag, grads[tag])
    print("4 ok  image gradient alive everywhere: "
          + ", ".join(f"{k} {v:.2e}" for k, v in grads.items()))


def test_psi_spread_breaks_the_cosine_rule():
    # one direction, 10° from e0 and therefore 80° from e1
    x = _at_angle(math.radians(10.0))
    u = torch.zeros(2, D); u[0, 0] = 1.0; u[1, 1] = 1.0
    cos = (F.normalize(x, dim=-1) @ u.T).squeeze()
    assert cos.argmax().item() == 0

    same = axis_cone_q(x, u * 4.0, K_R).squeeze()
    assert same.argmin().item() == 0, same

    # anchor 0 deep (narrow cone), anchor 1 shallow (wide cone)
    mixed = axis_cone_q(x, torch.stack([u[0] * 40.0, u[1] * 1.6]), K_R).squeeze()
    assert mixed.argmin().item() == 1, mixed
    print(f"5 ok  equal ψ → argmin q == argmax cos ({same[0]:.2f} vs {same[1]:.2f}); "
          f"unequal ψ → they diverge ({mixed[0]:.2f} vs {mixed[1]:.2f}, "
          f"cos {cos[0]:.3f} vs {cos[1]:.3f})")


def _channels(a_val, x, labels, lambda_neg):
    """(radial, tangential) autograd components of dL/da[0], and the same by finite
    difference. Any detach in the normalisation shows up as a mismatch."""
    loss_fn = AxisConeLoss(min_radius=K_R, lambda_neg=lambda_neg)
    a = a_val.clone().requires_grad_(True)
    loss, _ = loss_fn(x, a, labels)
    loss.backward()
    g = a.grad[0]

    u_hat = F.normalize(a_val[0], dim=-1)
    tang_dir = torch.zeros(D); tang_dir[2] = 1.0
    tang_dir = F.normalize(tang_dir - (tang_dir @ u_hat) * u_hat, dim=-1)

    def fd(direction, h=1e-4):
        out = []
        for sign in (+1, -1):
            pert = a_val.clone()
            pert[0] = pert[0] + sign * h * direction
            out.append(loss_fn(x, pert, labels)[0].item())
        return (out[0] - out[1]) / (2 * h)

    return ((g @ u_hat).item(), fd(u_hat),
            (g @ tang_dir).item(), fd(tang_dir))


def test_both_anchor_channels_are_alive():
    u = torch.zeros(2, D); u[0, 0] = 1.0; u[1, 1] = 1.0
    a_val = u * 4.0
    psi = math.asin(2 * K_R / 4.0)
    x = _at_angle(psi * 0.6, off=2)              # inside anchor 0's cone, off both axes

    # positive: label 0
    rad, rad_fd, tan, tan_fd = _channels(a_val, x, torch.zeros(1, dtype=torch.long), 0.0)
    assert abs(rad - rad_fd) < 1e-3 * max(1.0, abs(rad_fd)), (rad, rad_fd)
    assert abs(tan - tan_fd) < 1e-3 * max(1.0, abs(tan_fd)), (tan, tan_fd)
    assert rad > 0, rad          # descent SHRINKS ‖a‖ ⇒ widens the cone
    assert abs(tan) > 0, tan
    print(f"6a ok  positive: radial {rad:+.4f} (fd {rad_fd:+.4f}, widens the cone), "
          f"tangential {tan:+.4f} (fd {tan_fd:+.4f})")

    # negative: label 1, so anchor 0 is a wrong cone that contains the image
    rad, rad_fd, tan, tan_fd = _channels(a_val, x, torch.ones(1, dtype=torch.long), 1.0)
    assert abs(rad - rad_fd) < 1e-3 * max(1.0, abs(rad_fd)), (rad, rad_fd)
    assert rad < 0, rad          # descent GROWS ‖a‖ ⇒ narrows the cone
    print(f"6b ok  negative inside its cone: radial {rad:+.4f} (fd {rad_fd:+.4f}, "
          f"narrows the cone)")

    # and the axis rotates the right way in each case
    for tag, lab, lam, want_closer in (("own class", 0, 0.0, True),
                                       ("intruder", 1, 1.0, False)):
        a = a_val.clone().requires_grad_(True)
        AxisConeLoss(min_radius=K_R, lambda_neg=lam)(
            x, a, torch.full((1,), lab, dtype=torch.long))[0].backward()
        before = F.normalize(a_val[0], dim=-1) @ F.normalize(x[0], dim=-1)
        after = F.normalize(a_val[0] - 1e-3 * a.grad[0], dim=-1) @ F.normalize(x[0], dim=-1)
        assert (after > before) == want_closer, (tag, before.item(), after.item())
    print("6c ok  axis turns toward its own class and away from an intruder")


def test_norm_clamp():
    lo, hi = 1.5, 3.5
    t = torch.stack([torch.tensor([0.2] + [0.0] * (D - 1)),
                     torch.tensor([9.0] + [0.0] * (D - 1))])
    n = t.norm(dim=-1, keepdim=True)
    t = t * (n.clamp(lo, hi) / n.clamp_min(1e-8))
    assert torch.allclose(t.norm(dim=-1), torch.tensor([lo, hi])), t.norm(dim=-1)
    # and the floor is what keeps sin psi off its clamp: ‖a‖ = sinh‖u‖ must exceed 2K
    assert math.sinh(lo) > 2 * K_R, (math.sinh(lo), 2 * K_R)
    print(f"7 ok  norm clamp bilateral → [{lo}, {hi}]; sinh({lo})={math.sinh(lo):.2f} "
          f"> 2K={2 * K_R} so sin ψ never saturates")


if __name__ == "__main__":
    test_monotone_and_one_sided()
    test_q_is_one_on_the_wall()
    test_q_is_depth_invariant()
    test_no_dead_gradient_anywhere()
    test_psi_spread_breaks_the_cosine_rule()
    test_both_anchor_channels_are_alive()
    test_norm_clamp()
    print("\nall axis-cone invariants hold")
