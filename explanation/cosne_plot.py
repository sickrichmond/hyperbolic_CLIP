"""Exact, tiled CO-SNE plot of non-DALL-E checkpoint image embeddings.

Uses the affinities of Guo, Guo & Yu (CVPR 2022) and their stagewise update:
Riemannian KL descent, then a separate Euclidean radius correction after 500
iterations. The paper's mean radius loss and its printed update differ by N;
we follow the update. The KL rate scales with dataset size; the radius rate
does not. No code is copied from the authors' repository
(yunhuiguo/CO-SNE, commit 796a43b), which has no repository-wide license.
Every pair contributes; tiling limits peak memory. The all-image run is
O(N^2) per iteration and may require multiple Slurm allocations. Resume with
--resume after a time-limit exit.
Diagnostics save the KL-only layout at iteration 500 and the final layout,
each with radius summaries and class-highlighted disk panels.
"""

import argparse
import json
import math
import random
import re
import signal
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from checkpoint_io import atomic_torch_save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--captions_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--semantics", nargs="+", default=[
        "COCO", "cat", "dog", "wild", "FFHQ", "celebahq", "bedroom",
        "church", "classroom", "ImageNet-1k",
    ])
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Deterministic cap per generator; omit for every image")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--row-block", type=int, default=64,
                   help="Rows per all-pairs tile; reduce if GPU memory is insufficient")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--lambda-kl", type=float, default=10.0)
    p.add_argument("--lambda-radius", type=float, default=0.01)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="KL learning rate (default: N / (20 * lambda-kl)); "
                        "lambda-radius sets the separate radius step size")
    return p.parse_args()


def _validate_args(args):
    if args.max_per_class is not None and args.max_per_class < 1:
        raise ValueError("--max-per-class must be positive")
    if args.batch_size < 1 or args.num_workers < 0 or args.row_block < 1:
        raise ValueError("batch size and row block must be positive; workers nonnegative")
    if args.perplexity <= 1 or args.gamma <= 0 or args.lambda_kl <= 0:
        raise ValueError("perplexity, gamma and lambda-kl must be positive")
    if args.lambda_radius < 0 or (args.learning_rate is not None and args.learning_rate <= 0):
        raise ValueError("lambda-radius must be nonnegative and learning-rate positive")


def _select_subset(dataset, cap, seed):
    if cap is None:
        return
    by_generator = {}
    for index, (_, generator, _) in enumerate(dataset.samples):
        by_generator.setdefault(generator, []).append(index)
    for generator, indices in by_generator.items():
        if len(indices) < cap:
            raise ValueError(f"{generator} has only {len(indices)} images, fewer than {cap}")
    selected = set()
    for generator, indices in by_generator.items():
        selected.update(random.Random(f"{seed}:{generator}").sample(indices, cap))
    dataset.samples = [row for index, row in enumerate(dataset.samples) if index in selected]


def _pair_d2(u, v, u_sq=None, v_sq=None):
    """Squared Poincaré geodesic distances in the unit ball, for two row sets."""
    if u_sq is None:
        u_sq = u.square().sum(-1)
    if v_sq is None:
        v_sq = v.square().sum(-1)
    euclidean_sq = (u_sq[:, None] + v_sq[None, :] - 2 * (u @ v.T)).clamp_min_(0)
    a = 1 - u_sq
    b = 1 - v_sq
    z = 1 + 2 * euclidean_sq / (a[:, None] * b[None, :])
    return torch.acosh(z.clamp_min_(1)).square(), euclidean_sq, z


@torch.no_grad()
def _bandwidths(x, perplexity, row_block, beta, log_norm, start_row=0,
                progress=None):
    """Find each row's exact perplexity bandwidth over all other N-1 points."""
    n = len(x)
    target = math.log(perplexity)
    x_sq = x.square().sum(-1)
    for start in range(start_row, n, row_block):
        end = min(start + row_block, n)
        d2, _, _ = _pair_d2(x[start:end], x, x_sq[start:end], x_sq)
        d2[torch.arange(end - start, device=x.device),
           torch.arange(start, end, device=x.device)] = torch.inf
        finite_d2 = torch.where(torch.isfinite(d2), d2, 0)
        b = torch.ones(end - start, dtype=x.dtype, device=x.device)
        lower = torch.zeros_like(b)
        upper = torch.full_like(b, torch.inf)
        for _ in range(60):
            logits = -b[:, None] * d2
            z = torch.logsumexp(logits, dim=1)
            probs = torch.exp(logits - z[:, None])
            entropy = z + b * (probs * finite_d2).sum(dim=1)
            too_broad = entropy > target
            lower = torch.where(too_broad, b, lower)
            upper = torch.where(too_broad, upper, b)
            b = torch.where(too_broad,
                            torch.where(torch.isinf(upper), b * 2, (b + upper) / 2),
                            (b + lower) / 2)
        beta[start:end] = b
        log_norm[start:end] = torch.logsumexp(-b[:, None] * d2, dim=1)
        if progress is not None:
            progress(end, beta, log_norm)
    return beta, log_norm


@torch.no_grad()
def _exact_step(x, y, beta, log_norm, gamma, lambda_kl, lambda_radius,
                learning_rate, row_block, add_radius):
    """One exact all-pairs CO-SNE step, with O(row_block * N) memory."""
    n = len(x)
    x_sq = x.square().sum(-1)
    y_sq = y.square().sum(-1)
    a = 1 - y_sq
    attract = torch.empty_like(y)
    repel = torch.empty_like(y)
    sum_w = torch.zeros((), dtype=x.dtype, device=x.device)
    sum_p = torch.zeros_like(sum_w)
    kl_partial = torch.zeros_like(sum_w)
    gamma_sq = gamma * gamma
    tiny = torch.finfo(x.dtype).tiny
    for start in range(0, n, row_block):
        end = min(start + row_block, n)
        local = torch.arange(end - start, device=x.device)
        global_rows = torch.arange(start, end, device=x.device)

        input_d2, _, _ = _pair_d2(x[start:end], x, x_sq[start:end], x_sq)
        input_d2[local, global_rows] = torch.inf
        forward = torch.exp(-beta[start:end, None] * input_d2
                            - log_norm[start:end, None])
        backward = torch.exp(-beta[None, :] * input_d2 - log_norm[None, :])
        p = (forward + backward) / (2 * n)

        output_d2, sq, z = _pair_d2(y[start:end], y, y_sq[start:end], y_sq)
        output_d2[local, global_rows] = 0
        sq[local, global_rows] = 0
        z[local, global_rows] = 1
        denom = output_d2 + gamma_sq
        w = gamma / denom
        w[local, global_rows] = 0
        sum_w += w.sum()
        sum_p += p.sum()
        kl_partial += (p * (p.clamp_min(tiny).log()
                             - w.clamp_min(tiny).log())).sum()

        # d(d_H^2)/du = 8 d_H / sqrt(z^2-1) / (a_u a_v)
        #                  * ((u-v) + ||u-v||^2 u/a_u).
        # The distance ratio tends to 1 when z tends to 1.
        ratio = torch.where(z > 1 + 1e-12,
                            output_d2.sqrt() / ((z - 1) * (z + 1)).sqrt(),
                            torch.ones_like(z))
        ratio[local, global_rows] = 0
        factor = 16 * ratio / (a[start:end, None] * a[None, :] * denom)
        direction = (y[start:end, None, :] - y[None, :, :]
                     + (sq / a[start:end, None])[..., None] * y[start:end, None, :])
        attract[start:end] = ((p * factor)[..., None] * direction).sum(dim=1)
        repel[start:end] = ((w * factor)[..., None] * direction).sum(dim=1)

    if not torch.isfinite(sum_w) or sum_w <= 0:
        raise FloatingPointError("CO-SNE output affinity normalization failed")
    kl_grad = lambda_kl * (attract - repel / sum_w) * a.square()[:, None] / 4
    delta = x_sq - y_sq
    radius_loss = delta.square().mean()
    radius_grad = torch.zeros_like(y)
    if add_radius:
        # Equation 14 is a separate Euclidean update, not the gradient of
        # the mean radius loss reported above (which would contain 1 / N).
        radius_grad = -4 * lambda_radius * delta[:, None] * y
    update = learning_rate * kl_grad + radius_grad
    if not torch.isfinite(update).all():
        raise FloatingPointError("Non-finite CO-SNE gradient; previous state is preserved")
    updated = y - update
    lengths = updated.norm(dim=1, keepdim=True)
    updated = updated / (lengths / (1 - 1e-6)).clamp_min(1)
    if not torch.isfinite(updated).all():
        raise FloatingPointError("Non-finite CO-SNE update; previous state is preserved")
    kl = kl_partial + sum_p * sum_w.log()
    return (updated, float(kl), float(radius_loss),
            learning_rate * float(torch.linalg.vector_norm(kl_grad)),
            float(torch.linalg.vector_norm(radius_grad)))


def _convergence(completed, kl, radius, kl_norm, radius_norm, best_score,
                 best_iteration):
    """Check both update terms after the radius stage has begun."""
    if completed <= 500 or completed % 50:
        return best_score, best_iteration, None
    score = kl + radius
    if score < best_score:
        best_score, best_iteration = score, completed
    if kl_norm <= 1e-7 and radius_norm <= 1e-7:
        return best_score, best_iteration, "small_update"
    if completed - best_iteration > 300:
        return best_score, best_iteration, "no_progress"
    return best_score, best_iteration, None


def _radius_summary(radii):
    quantiles = np.quantile(radii, [0, 0.01, 0.5, 0.99, 1])
    return {
        **dict(zip(("min", "p01", "median", "p99", "max"), map(float, quantiles))),
        "mean": float(np.mean(radii)),
        "fraction_ge_0.99": float(np.mean(radii >= 0.99)),
        "fraction_ge_0.999": float(np.mean(radii >= 0.999)),
    }


def _plot(y, labels, path, completed=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    label_array = np.asarray(labels)
    names = list(dict.fromkeys(labels))
    colors = plt.colormaps["hsv"](np.linspace(0, 1, len(names), endpoint=False))
    for name, color in zip(names, colors):
        mask = label_array == name
        ax.scatter(y[mask, 0], y[mask, 1], label=name, color=color,
                   s=2, alpha=0.3, rasterized=True)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
    iteration = "" if completed is None else f"; iteration {completed}"
    ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02),
           title=f"CO-SNE of hyperbolic image embeddings ({len(y)} images{iteration})")
    ax.set_aspect("equal")
    ax.legend(markerscale=3, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_classes(y, labels, path, completed):
    label_array = np.asarray(labels)
    names = list(dict.fromkeys(labels))
    colors = plt.colormaps["hsv"](np.linspace(0, 1, len(names), endpoint=False))
    columns = min(4, len(names))
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.5 * rows),
                             squeeze=False)
    for ax, name, color in zip(axes.flat, names, colors):
        mask = label_array == name
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.75",
                               linewidth=0.6, linestyle="--", zorder=0))
        ax.scatter(y[~mask, 0], y[~mask, 1], color="0.65", s=0.5,
                   alpha=0.1, rasterized=True, zorder=1)
        ax.scatter(y[mask, 0], y[mask, 1], color=color, s=2,
                   alpha=0.65, rasterized=True, zorder=2)
        ax.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02),
               xticks=[-1, 0, 1], yticks=[-1, 0, 1],
               title=f"{name} ({int(mask.sum()):,})", aspect="equal")
    for ax in list(axes.flat)[len(names):]:
        ax.set_axis_off()
    fig.suptitle(f"CO-SNE class highlights ({len(y):,} images; iteration {completed})\n"
                 "Grey points show the other classes")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_diagnostics(prefix, y, manifest, settings, completed, input_radii,
                      stop_reason=None):
    coordinates = y.detach().cpu().numpy()
    output_radii = np.linalg.norm(coordinates, axis=1)
    summary = {
        "settings": settings, "completed": completed, "stop_reason": stop_reason,
        "radius_stage_iterations": max(0, completed - 500),
        "input": _radius_summary(input_radii),
        "output": _radius_summary(output_radii),
        "radius_mae": float(np.mean(np.abs(input_radii - output_radii))),
        "squared_radius_mse": float(np.mean((input_radii**2 - output_radii**2)**2)),
    }
    atomic_torch_save({"settings": settings, "coordinates": y.detach().cpu(),
                       "manifest": manifest, "completed": completed,
                       "stop_reason": stop_reason}, Path(f"{prefix}.points.pt"))
    Path(f"{prefix}.radii.json").write_text(json.dumps(summary, indent=2) + "\n")
    radii = summary["output"]
    print(f"Iteration {completed} output radii: min={radii['min']:.8f} "
          f"median={radii['median']:.8f} max={radii['max']:.8f} "
          f"fraction>=0.999={radii['fraction_ge_0.999']:.1%} "
          f"radius_MAE={summary['radius_mae']:.8f}", flush=True)
    labels = [gen for _, gen, _ in manifest]
    _plot(coordinates, labels, Path(f"{prefix}.png"), completed)
    _plot_classes(coordinates, labels, Path(f"{prefix}.classes.png"), completed)
    print(f"Saved diagnostics: {prefix}.{{points.pt,radii.json,png,classes.png}}", flush=True)


def main():
    args = parse_args()
    _validate_args(args)

    from torch.utils.data import DataLoader

    from data.iab_clip_dataset import IABCLIPDataset
    from models.attribution_clip import AttributionCLIP
    from training.poincare import extract_embeddings, lorentz_to_poincare

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving at the next safe point", flush=True)

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    ckpt_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem.removeprefix("attribution_22cls_")
    selection = "all" if args.max_per_class is None else f"perclass{args.max_per_class}"
    base_prefix = output_dir / f"cosne_plot_{stem}_{selection}_seed{args.seed}"
    embedding_path = Path(f"{base_prefix}.embeddings.pt")
    prefix = Path(f"{base_prefix}.cosne-v3")
    affinity_path = Path(f"{prefix}.affinities.pt")
    state_path = Path(f"{prefix}.state.pt")
    stage1_prefix = Path(f"{prefix}.iter500")
    if (args.resume and not state_path.exists() and not affinity_path.exists()
            and any(Path(f"{base_prefix}{suffix}").exists() for suffix in
                    (".state.pt", ".cosne-v2.state.pt"))):
        raise ValueError("Older CO-SNE optimizer state cannot resume as cosne-v3; start a new run")
    if state_path.exists() and not args.resume:
        raise FileExistsError(f"Existing optimizer state: {state_path}; use --resume")
    ckpt_stat = ckpt_path.stat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    curv = ckpt.get("curv", 1.0)
    if curv <= 0:
        raise ValueError(f"Invalid checkpoint curvature: {curv}")

    root = Path(args.dataset_path).resolve()
    generators = sorted(
        (p.name for p in root.iterdir() if p.is_dir()
         and not re.sub(r"[^a-z0-9]", "", p.name.lower()).startswith("dalle")),
        key=lambda name: (name != "real", name),
    )
    if not generators:
        raise ValueError(f"No non-DALL-E generator directories in {root}")
    dataset = IABCLIPDataset(
        root=str(root), captions_dir=args.captions_dir, generators=generators,
        semantics=args.semantics, processor_name=ckpt["clip_name"], split="all",
        seed=args.seed, require_caption=False,
    )
    _select_subset(dataset, args.max_per_class, args.seed)
    manifest = [(str(path), gen, sem) for path, gen, sem in dataset.samples]
    n = len(manifest)
    if n <= args.perplexity:
        raise ValueError(f"Need more than perplexity={args.perplexity} images; got {n}")
    signature = {
        "checkpoint": str(ckpt_path), "checkpoint_size": ckpt_stat.st_size,
        "checkpoint_mtime_ns": ckpt_stat.st_mtime_ns, "dataset": str(root),
        "captions_dir": str(Path(args.captions_dir).resolve()),
        "semantics": tuple(args.semantics), "seed": args.seed,
        "max_per_class": args.max_per_class, "curv": curv,
    }

    if embedding_path.exists():
        cached = torch.load(embedding_path, map_location="cpu", weights_only=False)
        if cached["signature"] != signature or cached["manifest"] != manifest:
            raise ValueError(f"Embedding cache does not match current input: {embedding_path}")
        x = cached["points"].to(device=device, dtype=torch.float64)
        print(f"Reusing {n} cached embeddings from {embedding_path}", flush=True)
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume without {embedding_path}")
        model = AttributionCLIP.from_checkpoint(ckpt).to(device)
        model.clip.load_state_dict(ckpt["lora_state"])
        model.projection.load_state_dict(ckpt["projection"])
        model.eval()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        embeddings, labels, semantics = extract_embeddings(model, loader, device)
        if labels != [gen for _, gen, _ in manifest] or semantics != [sem for _, _, sem in manifest]:
            raise RuntimeError("Embedding/manifest row order differs")
        points = lorentz_to_poincare(embeddings, curv)
        x = torch.as_tensor(points, dtype=torch.float64, device=device)
        if not torch.isfinite(x).all() or (x.square().sum(-1) >= 1).any():
            raise ValueError("Checkpoint embeddings are not finite points inside the unit ball")
        atomic_torch_save({"signature": signature, "manifest": manifest,
                           "points": x.cpu().float()}, embedding_path)
        del model, loader, embeddings, points
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("Embedding cache has an unexpected shape")
    if not torch.isfinite(x).all() or (x.square().sum(-1) >= 1).any():
        raise ValueError("Cached embeddings are not finite points inside the unit ball")
    del ckpt, dataset

    input_radii = x.norm(dim=1).detach().cpu().numpy()
    radii = _radius_summary(input_radii)
    print(f"Input radii: min={radii['min']:.8f} median={radii['median']:.8f} "
          f"max={radii['max']:.8f} fraction>=0.999={radii['fraction_ge_0.999']:.1%}",
          flush=True)

    learning_rate = args.learning_rate or n / (20 * args.lambda_kl)
    settings = {"algorithm": "cosne-v3", "signature": signature,
                "perplexity": args.perplexity,
                "gamma": args.gamma, "lambda_kl": args.lambda_kl,
                "lambda_radius": args.lambda_radius, "learning_rate": learning_rate,
                "n": n}
    beta = torch.empty(n, dtype=x.dtype, device=device)
    log_norm = torch.empty_like(beta)
    start_row = 0
    if args.resume and affinity_path.exists():
        affinity = torch.load(affinity_path, map_location="cpu", weights_only=False)
        if affinity["settings"] != settings:
            raise ValueError("Affinity checkpoint settings differ from this run")
        beta = affinity["beta"].to(device)
        log_norm = affinity["log_norm"].to(device)
        start_row = affinity["completed_rows"]
        if not 0 <= start_row <= n or beta.shape != (n,) or log_norm.shape != (n,):
            raise ValueError("Affinity checkpoint has an invalid shape or row count")

    if start_row < n:
        print(f"Computing exact input affinities: rows {start_row}/{n}", flush=True)

        def save_affinities(end, b, z):
            if end % 1024 < args.row_block or end == n or stop_requested:
                atomic_torch_save({"settings": settings, "completed_rows": end,
                                   "beta": b.cpu(), "log_norm": z.cpu()}, affinity_path)
                print(f"  affinity rows {end}/{n}", flush=True)
            if stop_requested:
                raise InterruptedError("Affinity search paused; resubmit with --resume")

        try:
            _bandwidths(x, args.perplexity, args.row_block, beta, log_norm,
                        start_row, save_affinities)
        except InterruptedError as exc:
            print(exc, flush=True)
            raise SystemExit(75) from None

    if args.resume and state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved.get("settings") != settings:
            raise ValueError("Optimizer checkpoint settings differ from this run")
        y = saved["points"].to(device)
        completed = saved["completed"]
        best_score = saved["best_score"]
        best_iteration = saved["best_iteration"]
        stop_reason = saved["stop_reason"]
        if y.shape != (n, 2) or not 0 <= completed <= 1000:
            raise ValueError("Optimizer checkpoint has an invalid shape or iteration")
        if not 0 <= best_iteration <= completed or stop_reason not in (
                None, "small_update", "no_progress", "max_iterations"):
            raise ValueError("Optimizer checkpoint has invalid convergence state")
    else:
        rng = torch.Generator(device="cpu").manual_seed(args.seed)
        y = (torch.randn((n, 2), generator=rng, dtype=x.dtype) * 0.01).to(device)
        completed = 0
        best_score, best_iteration, stop_reason = math.inf, 0, None

    if completed == 500:
        _save_diagnostics(stage1_prefix, y, manifest, settings, completed, input_radii)
    elif completed > 500 and not Path(f"{stage1_prefix}.points.pt").exists():
        print(f"Iteration-500 snapshot unavailable: resuming from iteration {completed}. "
              "A new run is needed to capture the KL-only layout.", flush=True)

    print(f"CO-SNE: {n} images, device={device}, row_block={args.row_block}, "
          f"learning_rate={learning_rate:g}, starting iteration={completed}", flush=True)
    if n > 100000:
        print(f"Exact mode evaluates {n * (n - 1):,} directed pairs per iteration; "
              "this may need many 24-hour jobs. Re-submit with --resume.", flush=True)
    for iteration in range(completed, 1000 if stop_reason is None else completed):
        started = time.monotonic()
        y, kl, radius, kl_norm, radius_norm = _exact_step(
            x, y, beta, log_norm, args.gamma, args.lambda_kl,
            args.lambda_radius, learning_rate, args.row_block,
            add_radius=iteration >= 500)
        completed = iteration + 1
        best_score, best_iteration, stop_reason = _convergence(
            completed, kl, radius, kl_norm, radius_norm,
            best_score, best_iteration)
        if completed == 1000 and stop_reason is None:
            stop_reason = "max_iterations"
        print(f"iteration {completed}/1000  KL={kl:.6f}  radius={radius:.6f}  "
              f"KL_update_norm={kl_norm:.6g}  radius_update_norm={radius_norm:.6g}  "
              f"seconds={time.monotonic() - started:.1f}", flush=True)
        if completed % 10 == 0 or stop_reason or stop_requested:
            atomic_torch_save({"settings": settings, "completed": completed,
                               "points": y.cpu(), "best_score": best_score,
                               "best_iteration": best_iteration,
                               "stop_reason": stop_reason}, state_path)
        if completed == 500:
            _save_diagnostics(stage1_prefix, y, manifest, settings, completed, input_radii)
        if stop_requested:
            print("CO-SNE paused; resubmit with --resume", flush=True)
            raise SystemExit(75)
        if stop_reason:
            print(f"CO-SNE stopped: {stop_reason} at iteration {completed}", flush=True)
            break

    _save_diagnostics(prefix, y, manifest, settings, completed, input_radii, stop_reason)


if __name__ == "__main__":
    main()
