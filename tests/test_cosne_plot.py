"""Dense checks for CO-SNE mean-loss descent and fixed iteration budget."""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checkpoint_io import atomic_torch_save
from explanation import cosne_plot
from explanation.cosne_plot import (
    _bandwidths, _exact_step, _pair_d2, _save_diagnostics, _select_subset,
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
        radius_grad = torch.autograd.grad(radius, z, retain_graph=True)[0]
        radius_grad *= (1 - y.square().sum(1))[:, None].square() / 4
        for block in (1, 3, n):
            for add_radius in (False, True):
                projected, tiled_kl, tiled_radius, kl_norm, radius_norm = _exact_step(
                    x, y, beta, log_norm, gamma, 10, 0.01, rate, block, add_radius)
                objective = 10 * kl + (0.01 * radius if add_radius else 0)
                grad = torch.autograd.grad(objective, z, retain_graph=True)[0]
                expected = rate * grad * (1 - y.square().sum(1))[:, None].square() / 4
                assert torch.allclose(y - projected, expected,
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
                           "settings": {"algorithm": "cosne-v4"}}, state)
        restored = torch.load(state, weights_only=True)
        assert restored["settings"]["algorithm"] == "cosne-v4"
        resumed, *_ = _exact_step(x, restored["points"], beta, log_norm,
                                  0.1, 10, 0.01, rate, 3, True)
    assert torch.equal(direct, resumed)


def test_radius_correction():
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
        for rate in (0.1, 1, 110):
            expected = y + (rate * 0.01 / n
                            * (x_radii.square() - y_radii.square())[:, None] * y
                            * (1 - y_radii.square())[:, None].square())
            updated, _, _, kl_norm, radius_norm = _exact_step(
                x, y, beta, log_norm, 0.1, 0, 0.01, rate, 2, True)
            assert torch.allclose(updated, expected, atol=1e-12)
            assert updated[0].norm() > y[0].norm()
            assert updated[1].norm() < y[1].norm()
            assert torch.equal(updated[2:], y[2:])
            assert kl_norm == 0 and radius_norm > 0


def test_main_runs_1000_iterations_and_resumes():
    # Flat losses and zero updates would have triggered the removed early stop.
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "data" / "real").mkdir(parents=True)
        ckpt = root / "checkpoint.pt"
        atomic_torch_save({"clip_name": "synthetic", "lora_state": {},
                           "projection": {}}, ckpt)
        manifest = [(str(root / "data" / "real" / f"{i}.png"), "real", "COCO")
                    for i in range(3)]
        modules = {
            "data.iab_clip_dataset": SimpleNamespace(
                IABCLIPDataset=lambda **kw: SimpleNamespace(samples=manifest.copy())),
            "models.attribution_clip": SimpleNamespace(AttributionCLIP=MagicMock()),
            "training.poincare": SimpleNamespace(
                extract_embeddings=lambda *a: (np.eye(3) * 0.1, ["real"] * 3, ["COCO"] * 3),
                lorentz_to_poincare=lambda embeddings, curv: embeddings),
        }
        argv = ["cosne_plot", "--checkpoint", str(ckpt), "--dataset_path", str(root / "data"),
                "--captions_dir", str(root / "captions"), "--output_dir", str(root),
                "--perplexity", "2", "--num_workers", "0"]
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                patch.object(cosne_plot.signal, "signal"), \
                patch.object(cosne_plot.torch.cuda, "is_available", return_value=False), \
                patch.object(cosne_plot, "_save_diagnostics") as diagnostics, \
                patch.object(cosne_plot, "_exact_step",
                             side_effect=lambda x, y, *a, **kw: (y, 1., 0.1, 0., 0.)) as step, \
                redirect_stdout(io.StringIO()):
            cosne_plot.main()
            assert step.call_count == 1000
            assert [call.kwargs["add_radius"] for call in step.call_args_list] == \
                [False] * 500 + [True] * 500
            assert all(call.args[7] == 3 / 200 for call in step.call_args_list)
            assert [call.args[4] for call in diagnostics.call_args_list] == [500, 1000]
            assert diagnostics.call_args.args[-1] == "max_iterations"
            state_path = next(root.glob("*.cosne-v4.state.pt"))
            state = torch.load(state_path, weights_only=True)
            assert state["completed"] == 1000 and state["settings"]["algorithm"] == "cosne-v4"
            step.reset_mock()
            argv.append("--resume")
            cosne_plot.main()
            step.assert_not_called()
            state["completed"] = 990
            atomic_torch_save(state, state_path)
            cosne_plot.main()
            assert step.call_count == 10
            assert torch.load(state_path, weights_only=True)["completed"] == 1000


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


def test_diagnostics_preserve_coordinates():
    coordinates = torch.tensor([[0., 0.], [0.3, 0.4], [-0.99995, 0.]],
                               dtype=torch.float64)
    original = coordinates.clone()
    input_radii = np.array([0.2, 0.5, 0.99995])
    original_radii = input_radii.copy()
    manifest = [(str(i), name, "COCO") for i, name in
                enumerate(("real", "other", "third"))]
    settings = {"algorithm": "cosne-v4", "lambda_radius": 0.01}
    with TemporaryDirectory() as directory:
        for completed, suffix, stop in ((500, ".iter500", None),
                                        (1000, "", "max_iterations")):
            prefix = Path(directory) / f"diagnostic{suffix}"
            _save_diagnostics(prefix, coordinates, manifest, settings,
                              completed, input_radii, stop)
            saved = torch.load(f"{prefix}.points.pt", weights_only=True)
            assert torch.equal(saved["coordinates"], original)
            assert saved["manifest"] == manifest and saved["settings"] == settings
            assert saved["completed"] == completed and saved["stop_reason"] == stop
            summary = json.loads(Path(f"{prefix}.radii.json").read_text())
            assert summary["radius_stage_iterations"] == max(0, completed - 500)
            assert summary["input"]["min"] == 0.2
            assert summary["output"]["median"] == 0.5
            assert summary["output"]["fraction_ge_0.999"] == 1 / 3
            assert abs(summary["radius_mae"] - 0.2 / 3) < 1e-12
            assert abs(summary["squared_radius_mse"] - 0.2**4 / 3) < 1e-12
            for suffix in (".png", ".classes.png"):
                assert Path(f"{prefix}{suffix}").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert torch.equal(coordinates, original)
    assert np.array_equal(input_radii, original_radii)


if __name__ == "__main__":
    test_tiled_step_matches_dense_autograd()
    test_radius_correction()
    test_main_runs_1000_iterations_and_resumes()
    test_coincident_points_and_boundary_projection()
    test_subset_is_balanced_and_reproducible()
    test_diagnostics_preserve_coordinates()
    print("CO-SNE checks passed")
