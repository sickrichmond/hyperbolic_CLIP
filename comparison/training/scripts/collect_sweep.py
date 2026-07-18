"""
Rank a HypCLIP hyperparameter sweep by balanced val accuracy.

Reads sweep_{i}.pt (each stores 'val_balanced') and pairs it with line i of the
configs file, printing configs best-first. Pick the winner, then retrain it longer.

    python comparison/training/scripts/collect_sweep.py \
        --dir $WORK/hyp_fine_tuning/checkpoints/sweep \
        --configs slurm/sweep_configs_22cls.txt
"""
import os
import glob
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, help="dir with sweep_*.pt")
    p.add_argument("--configs", required=True, help="sweep config list (one per line)")
    args = p.parse_args()

    lines = open(args.configs).read().splitlines()
    rows = []
    for f in glob.glob(os.path.join(args.dir, "sweep_*.pt")):
        try:
            i = int(os.path.basename(f)[len("sweep_"):-len(".pt")])
        except ValueError:
            continue
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[skip] {f}: {type(e).__name__}: {e}")
            continue
        vb = ck.get("val_balanced")
        cfg = lines[i] if i < len(lines) else "(config line missing)"
        rows.append((vb if vb is not None else -1.0, i, ck.get("epoch"), cfg))

    rows.sort(reverse=True)
    print(f"{'rank':>4}  {'task':>4}  {'val_bal':>8}  {'epoch':>5}  config")
    for r, (vb, i, ep, cfg) in enumerate(rows, 1):
        print(f"{r:>4}  {i:>4}  {100*vb:>7.2f}%  {str(ep):>5}  {cfg}")
    if rows:
        best = rows[0]
        print(f"\nBEST: task {best[1]}  val_balanced={100*best[0]:.2f}%\n  {best[3]}")
        print("→ retrain this config with --num_epochs 5 (or more) for the final model.")


if __name__ == "__main__":
    main()
