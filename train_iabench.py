"""Train the shared cone classifier on IABench's saved Arrow dataset.

Default partitions: 80% train, 10% validation, 10% test, stratified by generator.
Validation selects checkpoints; test rows are used only by the comparison CLI.
Per-epoch full-training plots are disabled; --diag_plot_dir still holds stats.csv.
Use --plot_all_train to request one final plot from the selected checkpoint.
"""
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

from data.iabench_dataset import IABenchDataset, manifest_digest
from train_attribution import run_training
from training.attribution_args import parse_args as shared_parse_args, validate_args


def parse_args(argv=None):
    return shared_parse_args(argv, dataset="iabench")


def prepare_datasets(args, metadata_only=False):
    dataset = IABenchDataset(args.dataset_path,
                            processor_name=None if metadata_only else args.clip_name)
    manifest = dataset.make_split_manifest(
        generators=args.generators, max_per_class=args.max_per_class, seed=args.seed,
        val_frac=args.val_frac, test_frac=args.test_frac)
    if len(manifest["class_names"]) < 2:
        raise ValueError("Training needs at least two generator classes")
    path = (Path(args.split_manifest) if args.split_manifest else
            Path(args.output).with_suffix(".splits.json"))
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != manifest:
            raise ValueError(f"{path} does not match dataset, classes, cap or split settings")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publication prevents an interrupted job leaving a partial manifest.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                             delete=False, encoding="utf-8") as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    args.generators = list(manifest["class_names"])
    args.split_manifest = str(path.resolve())
    print(f"IABench manifest: {path} — "
          + ", ".join(f"{s}={len(manifest[s])}" for s in ("train", "val", "test")))
    train = dataset.split_view(manifest, "train")
    val = dataset.split_view(manifest, "val")
    metadata = {"dataset": "iabench", "split_manifest": args.split_manifest,
                "split_manifest_digest": manifest_digest(manifest),
                "dataset_digest": manifest["dataset_digest"],
                "split_config": {key: manifest[key] for key in
                                 ("seed", "val_frac", "test_frac", "max_per_class")}}
    return (train, val), metadata


def main():
    args = parse_args()
    validate_args(args)
    start = perf_counter()
    datasets, metadata = prepare_datasets(args, metadata_only=args.prepare_only)
    print(f"IABench dataset/manifest preparation: {perf_counter() - start:.1f}s", flush=True)
    if args.prepare_only:
        print(f"Prepared split manifest: {args.split_manifest}")
        return
    run_training(args, datasets=datasets, checkpoint_metadata=metadata,
                 plot_each_epoch=False)


if __name__ == "__main__":
    main()
