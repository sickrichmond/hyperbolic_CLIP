"""CPU checks for entailment-cone hinges and anchor-norm regularization.

Run: python -m tests.test_phase_b
"""
import torch

from geometry.lorentz import exp_map0, half_aperture, oxy_angle
from losses.attribution_loss import EntailmentConeLoss, _subsample, predict_class

D, K, B = 8, 4, 6


def _ray(direction, norm):
    return direction / direction.norm(dim=-1, keepdim=True) * norm


def test_hinges_and_norm_match_formula():
    """Check values and gradients against direct positive/negative pair formulas."""
    for curv in (0.5, 1.0, 2.0):
        for mode in ("floor", "bilateral"):
            # Overlapping anchor directions exercise the wrong-class hinge too.
            anchors = _ray(torch.tensor([[1., .01], [1., -.01]]), 3).requires_grad_()
            images = _ray(torch.tensor([[1., .02], [1., .04], [1., .3]]), 9).requires_grad_()
            labels = torch.tensor([0, 1, 0])
            criterion = EntailmentConeLoss(
                curv=curv, min_radius=0.5, margin=0.3, lambda_neg=0.7,
                lambda_norm=0.4, target_norm=4, norm_mode=mode)
            loss, stats = criterion(images, anchors, labels)
            psi = half_aperture(anchors, curv=curv, min_radius=0.5)
            angles = torch.stack([
                oxy_angle(a.expand_as(images), images, curv=curv) for a in anchors
            ], dim=1)
            positive = (angles[torch.arange(3), labels] - psi[labels]).clamp_min(0).mean()
            mask = torch.arange(2).unsqueeze(0) != labels.unsqueeze(1)
            negative = (psi.unsqueeze(0) + 0.3 - angles).clamp_min(0)[mask].mean()
            deviation = 4 - anchors.norm(dim=-1)
            norm = (deviation.clamp_min(0) if mode == "floor" else deviation).square().mean()
            expected = positive + 0.7 * negative + 0.4 * norm
            assert positive > 0 and negative > 0
            assert torch.allclose(loss, expected)
            actual_grad = torch.autograd.grad(loss, (images, anchors), retain_graph=True)
            expected_grad = torch.autograd.grad(expected, (images, anchors))
            for actual, reference in zip(actual_grad, expected_grad):
                assert torch.isfinite(actual).all()
                assert torch.allclose(actual, reference, atol=1e-6)
            assert all(not v.requires_grad for v in stats.values())
            assert not list(criterion.parameters())
            assert torch.equal(predict_class(images, anchors, curv), angles.argmin(1))
    print("hinge/norm values, gradients, diagnostics and inference: OK")


def test_positive_hinge_stops_inside():
    anchors = _ray(torch.eye(K, D), 3)
    images = _ray(torch.eye(K, D) + 0.02 * torch.eye(K, D).roll(K, dims=1), 9)
    images.requires_grad_()
    labels = torch.arange(K)
    loss, stats = EntailmentConeLoss(min_radius=0.5, lambda_neg=0)(images, anchors, labels)
    assert stats["inside_img"] == 1
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(images.grad) == 0
    print("positive hinge is flat inside the cone: OK")


def test_norm_mode():
    x_img = exp_map0(torch.randn(B, D, generator=torch.Generator().manual_seed(1)))
    labels = torch.arange(B) % K
    deep = _ray(torch.eye(K, D), 8.18)     # spatial norm above the target

    floor = EntailmentConeLoss(min_radius=0.1, lambda_norm=1.0, target_norm=4.0)
    both = EntailmentConeLoss(min_radius=0.1, lambda_norm=1.0, target_norm=4.0,
                              norm_mode="bilateral")
    _, st_floor = floor(x_img, deep, labels)
    _, st_both = both(x_img, deep, labels)
    assert st_floor["loss_norm"].item() == 0.0, st_floor["loss_norm"]
    assert abs(st_both["loss_norm"].item() - (8.18 - 4.0) ** 2) < 1e-3, st_both["loss_norm"]
    print(f"4 ok  norm: floor 0.0 at depth 8.18, bilateral {st_both['loss_norm'].item():.3f}")


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

    # Check the curvature-1 Lorentz/Poincare coordinate round trip.
    x = exp_map0(torch.randn(7, D, generator=torch.Generator().manual_seed(3)))
    x_time = torch.sqrt(1.0 + (x ** 2).sum(-1, keepdim=True))
    p = x / (x_time + 1.0)
    back = 2 * p / (1 - (p ** 2).sum(-1, keepdim=True))
    assert torch.allclose(back, x, atol=1e-5), (back - x).abs().max()
    print("8 ok  norm clamp bilateral, Poincare round trip exact")


if __name__ == "__main__":
    test_hinges_and_norm_match_formula()
    test_positive_hinge_stops_inside()
    test_norm_mode()
    test_negative_subsample()
    test_anchor_clamp_and_poincare_round_trip()
    print("All hinge-loss checks passed.")
