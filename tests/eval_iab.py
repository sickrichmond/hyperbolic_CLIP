"""
Evaluate a trained step-1 checkpoint on ImageAttributionBench images.

Computes:
  - Norm statistics per class (real vs generator)
  - Threshold sweep: finds the threshold that maximises balanced accuracy
  - Per-generator breakdown at the best threshold

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m tests.eval_iab \\
        --checkpoint step1_checkpoint_large.pt \\
        --dataset_path /mnt/data3/rtrebiani/iab_dataset \\
        --model_classes FLUX real \\
        --semantic_classes COCO

To compare both checkpoints in one shot:
    for ckpt in step1_checkpoint_large.pt step1_checkpoint_paired.pt; do
        CUDA_VISIBLE_DEVICES=0 python -m tests.eval_iab \\
            --checkpoint $ckpt \\
            --dataset_path /mnt/data3/rtrebiani/iab_dataset
    done
"""
import argparse
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

from models.clip_lora import HyperbolicCLIP
from data.iab_dataset import IABDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a step-1 .pt checkpoint.")
    p.add_argument("--dataset_path", required=True,
                   help="Root of the extracted IAB dataset.")
    p.add_argument("--model_classes", nargs="+", default=["FLUX", "real"])
    p.add_argument("--semantic_classes", nargs="+", default=["COCO"])
    p.add_argument("--max_per_class", type=int, default=None,
                   help="Cap images per (generator, semantic) pair.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def threshold_sweep(dist: torch.Tensor, is_real: torch.Tensor,
                    n_steps: int = 200) -> tuple[float, float]:
    """Return (best_threshold, best_balanced_accuracy)."""
    lo, hi = dist.min().item(), dist.max().item()
    thresholds = torch.linspace(lo, hi, n_steps)
    best_t, best_ba = thresholds[0].item(), 0.0
    real_d = dist[is_real]
    fake_d = dist[~is_real]
    for t in thresholds:
        tpr = (real_d < t).float().mean().item()   # real correctly below threshold
        tnr = (fake_d >= t).float().mean().item()  # fake correctly above threshold
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t.item()
    return best_t, best_ba


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HyperbolicCLIP(hyperbolic_dim=128, curv=ckpt["curv"]).to(device)
    model.clip.load_state_dict(ckpt["lora_state"])
    model.projection.load_state_dict(ckpt["projection_state"])
    model.eval()

    dataset = IABDataset(
        root=args.dataset_path,
        generators=args.model_classes,
        semantics=args.semantic_classes,
        max_per_class=args.max_per_class,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_dist, all_real, all_gen = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="extracting embeddings"):
            pixel = batch["pixel_values"].to(device)
            _, tangent = model(pixel)
            all_dist.append(tangent.norm(dim=-1).cpu())
            all_real.append(batch["is_real"])
            all_gen.extend(batch["generator"])

    dist = torch.cat(all_dist)
    is_real = torch.cat(all_real)

    real_d = dist[is_real]
    fake_d = dist[~is_real]

    print(f"\n=== Norm statistics ({args.checkpoint}) ===")
    print(f"Real:  mean={real_d.mean():.3f}  std={real_d.std():.3f}  "
          f"min={real_d.min():.3f}  max={real_d.max():.3f}")
    print(f"Fake:  mean={fake_d.mean():.3f}  std={fake_d.std():.3f}  "
          f"min={fake_d.min():.3f}  max={fake_d.max():.3f}")
    print(f"Norm gap (fake_mean - real_mean): {fake_d.mean() - real_d.mean():.3f}")

    # ── Fixed threshold (1.25, same as legacy eval) ──────────────────────────
    t_fixed = 1.25
    pred_real_fixed = dist < t_fixed
    acc_fixed = (pred_real_fixed == is_real).float().mean().item()
    r2f_fixed = (real_d >= t_fixed).float().mean().item()
    f2r_fixed = (fake_d < t_fixed).float().mean().item()
    print(f"\n=== Threshold 1.25 (legacy) ===")
    print(f"Accuracy:              {100*acc_fixed:.1f}%  "
          f"({int((pred_real_fixed == is_real).sum())}/{len(is_real)})")
    print(f"Real → fake (FPR):     {100*r2f_fixed:.1f}%")
    print(f"Fake → real (FNR):     {100*f2r_fixed:.1f}%")

    # ── Best threshold (sweep) ────────────────────────────────────────────────
    best_t, best_ba = threshold_sweep(dist, is_real)
    pred_real_best = dist < best_t
    acc_best = (pred_real_best == is_real).float().mean().item()
    r2f_best = (real_d >= best_t).float().mean().item()
    f2r_best = (fake_d < best_t).float().mean().item()
    print(f"\n=== Best threshold (sweep over {len(dist)} samples) ===")
    print(f"Best threshold:        {best_t:.4f}")
    print(f"Balanced accuracy:     {100*best_ba:.1f}%")
    print(f"Accuracy:              {100*acc_best:.1f}%")
    print(f"Real → fake (FPR):     {100*r2f_best:.1f}%")
    print(f"Fake → real (FNR):     {100*f2r_best:.1f}%")

    # ── Per-generator breakdown (at best threshold) ───────────────────────────
    fake_by_gen: dict[str, list[float]] = {}
    for d, real, gen in zip(dist.tolist(), is_real.tolist(), all_gen):
        if not real:
            fake_by_gen.setdefault(gen, []).append(d)

    if fake_by_gen:
        print(f"\n=== Per-generator detection (fake, threshold {best_t:.3f}) ===")
        for gen, ds in sorted(fake_by_gen.items()):
            ds_t = torch.tensor(ds)
            detected = (ds_t >= best_t).sum().item()
            print(f"  {gen:20s}: n={len(ds):5d}  "
                  f"detected={detected}/{len(ds)} ({100*detected/len(ds):.0f}%)  "
                  f"mean_dist={ds_t.mean():.3f}")


if __name__ == "__main__":
    main()
