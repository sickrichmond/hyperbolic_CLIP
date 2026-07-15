"""
Dump the exact val/test image paths of the baseline stratified split, so the
hyperbolic model can be TRAINED excluding them (no leakage) while being evaluated
on byte-identical test images.

Reuses `get_dataloader` — the very function `test.py` uses — so the partition is
guaranteed identical to what the baselines are tested on (stratified by label,
seed 42, 2000/semantic cap). Paths are stored RELATIVE to root_dir (the on-disk
layout is shared between the harness dataset and data/iab_clip_dataset.py:
`{generator}/{super}/{sub}/{file}`), so training-time exclusion is robust to the
absolute root differing between machines.

Usage (on CINECA, once):
    python -m comparison.training.scripts.dump_split_manifest \\
        --root_dir $FAST/datasets/iab_dataset \\
        --out      $WORK/hyp_fine_tuning/split_manifest_default.json
"""
import os
import json
import argparse

from comparison.dataset.ImageAttributionDataset.dataloader import get_dataloader
from comparison.dataset.ImageAttributionDataset.semantic_split import get_semantic


def rel_paths(loader, root_dir):
    subset = loader.dataset            # torch Subset
    full = subset.dataset              # the (deep-copied) full dataset
    return sorted(os.path.relpath(full.samples[i][0], root_dir) for i in subset.indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model_name', default='resnet50',
                   help="Any baseline dataset works — all share the same enumeration, "
                        "so the split is identical. resnet50 avoids a CLIP download.")
    p.add_argument('--num_images_per_semantic_per_class', '-n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_semantic_split', action='store_true', default=False)
    p.add_argument('--task_id', type=int, default=1)
    args = p.parse_args()

    train_semantics = test_semantics = None
    if args.use_semantic_split:
        train_semantics, test_semantics = get_semantic(args.task_id)

    train_loader, val_loader, test_loader = get_dataloader(
        root_dir=args.root_dir,
        model_name=args.model_name,
        num_images_per_semantic_per_class=args.num_images_per_semantic_per_class,
        batch_size=1,
        degraded=0,
        config={'model_name': args.model_name},
        num_workers=0,
        seed=args.seed,
        use_semantic_split=args.use_semantic_split,
        train_semantics=train_semantics,
        test_semantics=test_semantics,
    )

    manifest = {
        'root_dir': args.root_dir,
        'seed': args.seed,
        'use_semantic_split': args.use_semantic_split,
        'task_id': args.task_id if args.use_semantic_split else None,
        'num_images_per_semantic_per_class': args.num_images_per_semantic_per_class,
        # paths RELATIVE to root_dir; exclude these from ours' training
        'val': rel_paths(val_loader, args.root_dir),
        'test': rel_paths(test_loader, args.root_dir),
        'n_train': len(train_loader.dataset),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(manifest, f)
    print(f"Wrote manifest → {args.out}")
    print(f"  train={manifest['n_train']}  val={len(manifest['val'])}  "
          f"test={len(manifest['test'])}")


if __name__ == '__main__':
    main()
