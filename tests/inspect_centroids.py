"""Inspect an anchor-centroid cache: how separable are the per-class means?

    python -m tests.inspect_centroids $WORK/hyp_fine_tuning/anchor_centroids_22cls.pt
    python -m tests.inspect_centroids $WORK/hyp_fine_tuning/anchor_centroids_22cls_spectral.pt

The centroids are the INIT of the free anchors, so two classes whose CLIP-space
means point almost the same way start with overlapping cones, and one can swallow
the other for good. Two open questions this answers in seconds instead of a 20h
job:

  - the spectral branch sat at 1/22 = chance accuracy for a whole run. If its
    off-diagonal cosines are ~0.99+, the FFT-into-CLIP embedding is near constant
    across classes and no training budget fixes that.
  - mid-6.0 stayed at 0.0% for five epochs while mid-5.2 was at 100%. If those two
    centroids are near-parallel, the collapse is in the initialisation.
"""
import sys

import torch
import torch.nn.functional as F


def report(path: str, top: int = 10) -> None:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    names, mean = blob["class_names"], F.normalize(blob["mean_clip"].float(), dim=-1)
    K = len(names)
    cos = mean @ mean.T
    off = cos[~torch.eye(K, dtype=torch.bool)]

    print(f"\n{path}\n  {K} classes, dim {mean.shape[1]}")
    print(f"  off-diagonal cosine: mean={off.mean():.4f}  min={off.min():.4f}  "
          f"max={off.max():.4f}")
    if off.mean() > 0.98:
        print("  ⚠️  centroids are nearly identical — this branch has almost no "
              "class signal to start from")

    pairs = sorted(
        ((cos[i, j].item(), names[i], names[j])
         for i in range(K) for j in range(i + 1, K)),
        reverse=True,
    )
    print(f"  {top} most similar pairs:")
    for c, a, b in pairs[:top]:
        print(f"    {c:.4f}  {a} ↔ {b}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"usage: python -m tests.inspect_centroids <cache.pt> [more.pt ...]")
    for p in sys.argv[1:]:
        report(p)
