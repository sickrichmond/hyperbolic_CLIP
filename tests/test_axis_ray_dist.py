"""CPU checks for distance to the geodesic ray beyond a cone apex.

Compare axis_ray_dist with brute-force minimization at three curvatures.
Check branch continuity, angular monotonicity, image gradients on both
branches, radial/tangential anchor gradients and finite on-axis gradients.

Run: python -m tests.test_axis_ray_dist
"""
import math

import torch

from geometry.lorentz import axis_ray_dist, elementwise_dist

# float64 throughout: checks 4-5 compare autograd against finite differences, and in
# float32 the difference quotients are a few hundred ulp of the distance — they would
# disagree at the 1e-2 level for pure rounding reasons, hiding a real discrepancy of the
# same size. Everything here is CPU and tiny, so it is free.
torch.set_default_dtype(torch.float64)

D = 6


def _pt(theta: float, r: float, axis: int = 0, off: int = 1) -> torch.Tensor:
    v = torch.zeros(1, D)
    v[0, axis] = r * math.cos(theta)
    v[0, off] = r * math.sin(theta)
    return v


def _brute(x: torch.Tensor, a: torch.Tensor, curv: float, n: int = 40000) -> float:
    """Minimise d_H(p, x) over p on the ray from a outward, by dense sampling."""
    rc = curv ** 0.5
    a_norm = a.norm().item()
    r_a = math.asinh(rc * a_norm) / rc
    a_hat = (a / a_norm).expand(n, D)
    t = r_a + torch.arange(n, dtype=torch.float64) * 2e-4
    s = (torch.sinh(rc * t) / rc).unsqueeze(1)
    return elementwise_dist(a_hat * s, x.expand(n, D), curv=curv).min().item()


def _branch_switch_angle(r_x: float, a_norm: float, curv: float = 1.0) -> float:
    """theta where x_par/x_time == ||a||/a_time, i.e. the perpendicular foot hits a."""
    x_time = math.sqrt(1 / curv + r_x ** 2)
    a_time = math.sqrt(1 / curv + a_norm ** 2)
    return math.acos((a_norm / a_time) * x_time / r_x)


def test_matches_brute_force():
    a_norm = 3.63
    worst = 0.0
    for curv in (1.0, 0.5, 2.0):
        a = _pt(0.0, a_norm)
        for th_deg in (0.5, 2, 5, 10, 20, 45, 80, 100, 160, 179.5):
            for r_x in (4.0, 10.0, 40.0):
                x = _pt(math.radians(th_deg), r_x)
                got = axis_ray_dist(x, a, curv=curv).item()
                worst = max(worst, abs(got - _brute(x, a, curv)))
    assert worst < 1e-6, f"disagrees with brute force by {worst:.2e}"
    print(f"1 ok  matches brute-force minimisation over the ray at curv 0.5/1/2 "
          f"(max err {worst:.1e})")


def test_continuous_at_the_branch_switch():
    a_norm, r_x = 3.63, 10.0
    a = _pt(0.0, a_norm)
    th = _branch_switch_angle(r_x, a_norm)
    vals = [axis_ray_dist(_pt(th + d, r_x), a).item()
            for d in (-1e-6, -1e-9, 1e-9, 1e-6)]
    jump = abs(vals[2] - vals[1])
    assert jump < 1e-8, f"discontinuous at the switch by {jump:.2e}"
    print(f"2 ok  continuous across the branch switch at theta={math.degrees(th):.2f} deg "
          f"(jump {jump:.1e}); the foot there IS the apex")


def test_strictly_monotone_over_the_whole_range():
    a = _pt(0.0, 3.63)
    for r_x in (4.0, 10.0, 40.0):
        d = [axis_ray_dist(_pt(i * math.pi / 2000, r_x), a).item()
             for i in range(1, 2000)]
        steps = [b - c for c, b in zip(d, d[1:])]
        assert min(steps) > 0, f"not monotone at ||x||={r_x}: min step {min(steps):.2e}"
        # the bilateral form the ray construction exists to avoid
        bad = [(math.sin(i * math.pi / 2000) * r_x) for i in range(1, 2000)]
        assert bad[-1] < bad[len(bad) // 2], "sanity: ||x_perp|| should fold back"
    print("3 ok  strictly monotone on all of [0, pi] at three depths; the full-geodesic "
          "form folds back past 90 deg")


def test_no_dead_gradient_on_either_branch():
    a_norm, r_x = 3.63, 10.0
    a = _pt(0.0, a_norm)
    th_sw = _branch_switch_angle(r_x, a_norm)
    for tag, th in (("perp", th_sw / 2), ("apex", (th_sw + math.pi) / 2)):
        x = _pt(th, r_x).requires_grad_(True)
        axis_ray_dist(x, a).sum().backward()
        g = x.grad.norm().item()
        assert g > 1e-6, f"{tag} branch has a dead gradient on the image: {g:.2e}"
        print(f"4 ok  {tag} branch (theta={math.degrees(th):6.2f} deg): "
              f"|grad_x| = {g:.4f}")
    # and near pi, where the ray-from-origin variant would be exactly flat
    x = _pt(math.pi - 1e-3, r_x).requires_grad_(True)
    axis_ray_dist(x, a).sum().backward()
    assert x.grad.norm().item() > 1e-6, "dead gradient at theta -> pi"
    print(f"4 ok  still alive at theta=179.94 deg: |grad_x| = {x.grad.norm():.4f}")


def test_anchor_moves_radially_only_on_the_apex_branch():
    a_norm, r_x = 3.63, 10.0
    th_sw = _branch_switch_angle(r_x, a_norm)
    out = {}
    for tag, th in (("perp", th_sw / 2), ("apex", (th_sw + math.pi) / 2)):
        a = _pt(0.0, a_norm).requires_grad_(True)
        axis_ray_dist(_pt(th, r_x), a).sum().backward()
        g = a.grad[0]
        a_hat = (a.detach()[0] / a.detach()[0].norm())
        radial = torch.dot(g, a_hat).abs().item()
        tangential = (g - torch.dot(g, a_hat) * a_hat).norm().item()
        out[tag] = (radial, tangential)
    assert out["perp"][0] < 1e-9, \
        f"perpendicular branch must not move the anchor radially: {out['perp'][0]:.2e}"
    assert out["perp"][1] > 1e-6, "perpendicular branch must rotate the anchor"
    assert out["apex"][0] > 1e-6, \
        f"apex branch must move the anchor radially: {out['apex'][0]:.2e}"
    print(f"5 ok  anchor gradient  perp: radial {out['perp'][0]:.2e} / tangential "
          f"{out['perp'][1]:.4f}  (rotates only)")
    print(f"5 ok  anchor gradient  apex: radial {out['apex'][0]:.4f} / tangential "
          f"{out['apex'][1]:.4f}  (depth is learnable)")


def test_on_axis_is_zero_with_a_finite_gradient():
    a = _pt(0.0, 3.63)
    x = _pt(0.0, 10.0).requires_grad_(True)          # exactly ON the axis
    d = axis_ray_dist(x, a)
    assert d.item() < 1e-7, f"on-axis distance should be 0, got {d.item():.2e}"
    d.sum().backward()
    assert torch.isfinite(x.grad).all(), f"NaN/Inf gradient on the axis: {x.grad}"
    print(f"6 ok  exactly on the axis: d = {d.item():.2e}, gradient finite "
          f"(|grad| = {x.grad.norm():.2e}) — torch.norm alone would give NaN here")


if __name__ == "__main__":
    test_matches_brute_force()
    test_continuous_at_the_branch_switch()
    test_strictly_monotone_over_the_whole_range()
    test_no_dead_gradient_on_either_branch()
    test_anchor_moves_radially_only_on_the_apex_branch()
    test_on_axis_is_zero_with_a_finite_gradient()
    print("\nall axis-ray invariants hold")
