"""
Fase B — Linear probe on FROZEN CLIP features.

This is the professor's baseline: it measures how much of the 22-way generator
attribution is ALREADY linearly decodable from off-the-shelf CLIP image features,
with NO LoRA fine-tuning and NO special geometry. If this probe already scores
~98%, then fine-tuning + cones + dimension are all second-order.

Input: the two caches written by scripts/extract_clip_features.py
    <features_dir>/clip_features_train.pt   {"X": (N,768), "y": (N,), "classes": [...]}
    <features_dir>/clip_features_val.pt

It trains a single nn.Linear(feat_dim -> num_classes) with class-balanced
cross-entropy (the train set is imbalanced: real has ~8800 captioned images vs
~16000 per generator), then reports overall / balanced / per-class accuracy and a
confusion matrix — same metrics as tests/eval_attribution, so the number lines up
directly against the hyperbolic and euclidean fine-tuned models.

There is no CLIP here: we operate on the cached 768-d features, so it trains in a
couple of minutes (seconds on a GPU).

Usage:
    python train_linear_probe.py \
        --features_dir $WORK/clip_features \
        --output       $WORK/checkpoints/linear_probe.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_dir", required=True,
                   help="Directory holding clip_features_{train,val}.pt")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--batch_size",    type=int,   default=4096)
    p.add_argument("--no_class_weight", action="store_true",
                   help="Disable class-balanced CE (plain cross-entropy).")
    p.add_argument("--eval_every",    type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output",        default="linear_probe.pt")
    return p.parse_args()


@torch.no_grad()
def evaluate(linear, X, y, num_classes, device):
    """Return overall acc, balanced acc, per-class recall list, confusion matrix."""
    linear.eval()
    preds = []
    for i in range(0, len(X), 8192):
        preds.append(linear(X[i:i + 8192].to(device)).argmax(1).cpu())
    pred = torch.cat(preds)

    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, pr in zip(y.tolist(), pred.tolist()):
        conf[t, pr] += 1
    totals = conf.sum(axis=1)
    recalls = [conf[c, c] / totals[c] if totals[c] else 0.0 for c in range(num_classes)]
    overall = (pred == y).float().mean().item()
    balanced = float(np.mean(recalls))
    return overall, balanced, recalls, conf


def print_report(classes, overall, balanced, recalls, conf, n_val, title):
    """Print overall / balanced / per-class accuracy + confusion matrix.
    Shared by train_linear_probe and eval_linear_probe."""
    print(f"\n=== {title} ===")
    print(f"Overall accuracy:  {100 * overall:.1f}%")
    print(f"Balanced accuracy: {100 * balanced:.1f}%   ({n_val} val samples)\n")
    print("--- Per-class accuracy (recall) ---")
    for c, r in zip(classes, recalls):
        print(f"  {c:16s}: {100 * r:5.1f}%")
    print("\n--- Confusion matrix (rows = ground truth, cols = prediction) ---")
    head = "".join(f"{c[:6]:>8}" for c in classes)
    print(f"{'':16}{head}")
    for c, row in zip(classes, conf):
        print(f"  gt={c:13s}" + "".join(f"{v:>8}" for v in row))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat = Path(args.features_dir)
    tr = torch.load(feat / "clip_features_train.pt", weights_only=False)
    va = torch.load(feat / "clip_features_val.pt",   weights_only=False)
    X_train, y_train, classes = tr["X"].float(), tr["y"].long(), list(tr["classes"])
    X_val,   y_val            = va["X"].float(), va["y"].long()
    num_classes = len(classes)
    feat_dim = X_train.shape[1]
    print(f"Loaded features: train {tuple(X_train.shape)}  val {tuple(X_val.shape)}  "
          f"classes={num_classes}  device={device}")

    # Class-balanced CE: weight_c = N / (K * count_c)  (downweights frequent classes)
    counts = torch.bincount(y_train, minlength=num_classes).float()
    weight = None if args.no_class_weight else \
        (counts.sum() / (num_classes * counts.clamp(min=1))).to(device)
    if weight is None:
        print("Class weighting: OFF (plain CE)")
    else:
        print(f"Class weighting: ON  (train counts min={int(counts.min())} "
              f"max={int(counts.max())})")

    linear = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.AdamW(linear.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss(weight=weight)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size,
                        shuffle=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_balanced = -1.0
    for epoch in range(1, args.epochs + 1):
        linear.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = ce(linear(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            overall, balanced, _, _ = evaluate(linear, X_val, y_val, num_classes, device)
            print(f"epoch {epoch:3d}  train_loss={running / len(X_train):.4f}  "
                  f"val_overall={100 * overall:.1f}%  val_balanced={100 * balanced:.1f}%")
            if balanced > best_balanced:
                best_balanced = balanced
                torch.save({"state_dict": linear.state_dict(), "classes": classes,
                            "feat_dim": feat_dim, "val_balanced": balanced,
                            "epoch": epoch}, out_path)

    # ── Final detailed report on the best checkpoint ─────────────────────────
    best = torch.load(out_path, weights_only=False)
    linear.load_state_dict(best["state_dict"])
    overall, balanced, recalls, conf = evaluate(linear, X_val, y_val, num_classes, device)

    print_report(classes, overall, balanced, recalls, conf, len(X_val),
                 f"Linear probe on FROZEN CLIP — best epoch {best['epoch']}")
    print(f"\nBest balanced val accuracy: {100 * best_balanced:.1f}%  ({out_path})")


if __name__ == "__main__":
    main()
