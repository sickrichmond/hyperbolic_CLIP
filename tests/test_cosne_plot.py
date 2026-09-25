"""Small dense checks for the exact tiled CO-SNE objective and gradient."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from checkpoint_io import atomic_torch_save
from explanation.cosne_plot import _bandwidths, _exact_step, _pair_d2, _select_subset


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
    w = 0.1 / (distance + 0.1**2)
    w = w * (1 - torch.eye(n, dtype=z.dtype))
    q = w / w.sum()
    kl = (p * (p.clamp_min(torch.finfo(p.dtype).tiny).log()
               - q.clamp_min(torch.finfo(q.dtype).tiny).log())).sum()
    radius = (x.square().sum(1) - z.square().sum(1)).square().mean()
    loss = 10 * kl + 0.01 * radius
    euclidean_grad = torch.autograd.grad(loss, z)[0]
    riemannian_grad = euclidean_grad * (1 - y.square().sum(1))[:, None].square() / 4

    rate = 0.001
    tiled_y, tiled_kl, tiled_radius = _exact_step(
        x, y, beta, log_norm, 0.1, 10, 0.01, rate, 3, True)
    assert torch.allclose((y - tiled_y) / rate, riemannian_grad,
                          rtol=1e-6, atol=1e-8)
    assert abs(tiled_kl - float(kl)) < 1e-9
    assert abs(tiled_radius - float(radius)) < 1e-12
    whole_y, _, _ = _exact_step(x, y, beta, log_norm, 0.1, 10, 0.01,
                                rate, n, True)
    assert torch.allclose(tiled_y, whole_y, rtol=1e-9, atol=1e-11)
    assert (tiled_y.norm(dim=1) < 1).all()
    _, _, early_radius = _exact_step(x, y, beta, log_norm, 0.1, 10, 0.01,
                                     rate, 3, False)
    assert early_radius == 0

    direct, _, _ = _exact_step(x, tiled_y, beta, log_norm, 0.1, 10, 0.01,
                               rate, 3, True)
    with TemporaryDirectory() as directory:
        state = Path(directory) / "state.pt"
        atomic_torch_save({"completed": 1, "points": tiled_y}, state)
        restored = torch.load(state, weights_only=True)
        resumed, _, _ = _exact_step(x, restored["points"], beta, log_norm,
                                    0.1, 10, 0.01, rate, 3, True)
    assert torch.equal(direct, resumed)


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
    test_subset_is_balanced_and_reproducible()
    print("CO-SNE checks passed")
