"""Report cosine similarities between cached per-class CLIP centroids.

Read class_names and mean_clip from an anchor-centroid cache, normalize the
means, and print off-diagonal statistics and the closest class pairs. This
describes class means, not individual-image separability or training outcomes.

Usage: python -m tests.inspect_centroids cache.pt [more.pt ...]
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
