"""Small dense checks for the tiled CO-SNE affinities and stagewise update."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from checkpoint_io import atomic_torch_save
from explanation.cosne_plot import (
    _bandwidths, _convergence, _exact_step, _pair_d2, _select_subset,
)


def test_tiled_step_matches_dense_autograd():
    torch.manual_seed(7)
    x = torch.randn(7, 4, dtype=torch.float64) * 0.12
    y = torch.randn(7, 2, dtype=torch.float64) * 0.08
    n = len(x)
    beta = torch.empty(n, dtype=x.dtype)
    log_norm = torch.empty_like(beta)
    _bandwidths(x, 2.5, 3, beta, log_norm)

    input_d2 = _pair_d2(x, x)[0]
    input_d2.fill_diagonal_(torch.inf)
    forward = torch.exp(-beta[:, None] * input_d2 - log_norm[:, None])
    p = (forward + forward.T) / (2 * n)
    assert torch.allclose(p.sum(), torch.tensor(1.0, dtype=x.dtype), atol=1e-10)
    assert torch.allclose(p, p.T)
    entropy = -(forward * forward.clamp_min(torch.finfo(x.dtype).tiny).log()).sum(1)
    assert torch.allclose(entropy.exp(), torch.full_like(entropy, 2.5), atol=1e-8)

    z = y.clone().requires_grad_(True)
    # Clamp only to make the self-distance's unused derivative finite.
    sq = (z.square().sum(1)[:, None] + z.square().sum(1)[None, :]
          - 2 * z @ z.T).clamp_min(0)
    norms = 1 - z.square().sum(1)
    distance = torch.acosh((1 + 2 * sq / (norms[:, None] * norms[None, :]))
                           .clamp_min(1 + 1e-12)).square()
    rate = 0.001
    for gamma in (0.1, 0.5):
        w = gamma / (distance + gamma**2)
        w = w * (1 - torch.eye(n, dtype=z.dtype))
        q = w / w.sum()
        kl = (p * (p.clamp_min(torch.finfo(p.dtype).tiny).log()
                   - q.clamp_min(torch.finfo(q.dtype).tiny).log())).sum()
        kl_grad = torch.autograd.grad(kl, z, retain_graph=True)[0]
        kl_grad *= (1 - y.square().sum(1))[:, None].square() / 4
        radius = (x.square().sum(1) - z.square().sum(1)).square().mean()
        radius_grad = -4 * (x.square().sum(1) - y.square().sum(1))[:, None] * y
        for block in (1, 3, n):
            for add_radius in (False, True):
                projected, tiled_kl, tiled_radius, kl_norm, radius_norm = _exact_step(
                    x, y, beta, log_norm, gamma, 10, 0.01, rate, block, add_radius)
                expected = 10 * kl_grad + (0.01 * radius_grad if add_radius else 0)
                assert torch.allclose((y - projected) / rate, expected,
                                      rtol=1e-6, atol=1e-8)
                assert abs(tiled_kl - kl.detach().item()) < 1e-9
                assert abs(tiled_radius - radius.detach().item()) < 1e-12
                assert abs(kl_norm - (rate * 10 * kl_grad).detach().norm().item()) < 1e-8
                assert abs(radius_norm - ((rate * 0.01 * radius_grad).norm().item()
                                          if add_radius else 0)) < 1e-12
                assert (projected.norm(dim=1) < 1).all()

    tiled_y, *_ = _exact_step(x, y, beta, log_norm, 0.1, 10, 0.01, rate, 3, True)
    direct, *_ = _exact_step(x, tiled_y, beta, log_norm, 0.1, 10, 0.01,
                             rate, 3, True)
    with TemporaryDirectory() as directory:
        state = Path(directory) / "state.pt"
        atomic_torch_save({"completed": 1, "points": tiled_y,
                           "best_score": float("inf"), "best_iteration": 0,
                           "stop_reason": None, "settings": {"algorithm": "cosne-v2"}}, state)
        restored = torch.load(state, weights_only=True)
        assert restored["settings"]["algorithm"] == "cosne-v2"
        resumed, *_ = _exact_step(x, restored["points"], beta, log_norm,
                                  0.1, 10, 0.01, rate, 3, True)
    assert torch.equal(direct, resumed)


def test_radius_correction_and_convergence():
    for n in (3, 9):
        angles = torch.arange(n, dtype=torch.float64) * (2 * torch.pi / n)
        directions = torch.stack((angles.cos(), angles.sin()), dim=1)
        x_radii = torch.full((n,), 0.3, dtype=torch.float64)
        y_radii = x_radii.clone()
        x_radii[:2] = torch.tensor([0.9, 0.2])
        y_radii[:2] = torch.tensor([0.2, 0.9])
        x, y = x_radii[:, None] * directions, y_radii[:, None] * directions
        d2 = _pair_d2(x, x)[0]
        d2.fill_diagonal_(torch.inf)
        beta = torch.ones(n, dtype=x.dtype)
        log_norm = torch.logsumexp(-d2, dim=1)
        updated, _, _, kl_norm, radius_norm = _exact_step(
            x, y, beta, log_norm, 0.1, 0, 0.01, 1, 2, True)
        expected = y + 4 * 0.01 * (x_radii.square() - y_radii.square())[:, None] * y
        assert torch.allclose(updated, expected, atol=1e-12)
        assert updated[0].norm() > y[0].norm()
        assert updated[1].norm() < y[1].norm()
        assert torch.equal(updated[2:], y[2:])
        assert kl_norm == 0 and radius_norm > 0
    assert _convergence(500, 1, 0.1, 0, 0, float("inf"), 0)[2] is None
    best, iteration, stop = _convergence(550, 1, 0.1, 1, 1, float("inf"), 0)
    assert (best, iteration, stop) == (1.1, 550, None)
    assert _convergence(600, 1, 0.1, 0, 0, best, iteration)[2] == "small_update"
    assert _convergence(900, 1, 0.1, 1, 1, best, iteration)[2] == "no_progress"


def test_coincident_points_and_boundary_projection():
    x = torch.tensor([[0.1, 0.0], [0.0, 0.2], [-0.2, 0.0]], dtype=torch.float64)
    y = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.999, 0.0]], dtype=torch.float64)
    d2 = _pair_d2(x, x)[0]
    d2.fill_diagonal_(torch.inf)
    beta = torch.ones(3, dtype=x.dtype)
    log_norm = torch.logsumexp(-d2, dim=1)
    updated, kl, radius, kl_norm, radius_norm = _exact_step(
        x, y, beta, log_norm, 0.1, 10, 0.01, 10, 1, True)
    assert torch.isfinite(updated).all()
    assert torch.tensor([kl, radius, kl_norm, radius_norm]).isfinite().all()
    assert (updated.norm(dim=1) < 1).all()


def test_subset_is_balanced_and_reproducible():
    class Dataset:
        def __init__(self):
            self.samples = [(f"{gen}/{i}.png", gen, "COCO")
                            for gen in ("real", "other") for i in range(10)]

    first, second = Dataset(), Dataset()
    _select_subset(first, 4, 42)
    _select_subset(second, 4, 42)
    assert first.samples == second.samples
    assert [g for _, g, _ in first.samples].count("real") == 4
    assert [g for _, g, _ in first.samples].count("other") == 4


if __name__ == "__main__":
    test_tiled_step_matches_dense_autograd()
    test_radius_correction_and_convergence()
    test_coincident_points_and_boundary_projection()
    test_subset_is_balanced_and_reproducible()
    print("CO-SNE checks passed")
