"""Collect the fixed 18-run IAB OmniDFA sweep; rank by clean validation only."""
import argparse
import csv
from itertools import product
import math
from pathlib import Path


CONDITIONS = {0: "clean", 4: "jpeg30", 6: "blur5"}
CONFIGS = tuple(product((0.2, 0.5), (2, 3, 4), (2, 4, 6)))
FIELDS = ("task", "z", "tau", "lambda_anchor", "epoch", "checkpoint", "val_balanced",
          *(f"{name}_{metric}" for name in CONDITIONS.values()
            for metric in ("acc", "balanced_accuracy")), "status", "error")


def accuracy(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid accuracy: {value}")
    return value


def collect(directory):
    import torch

    rows = []
    for task, (weight, z, tau) in enumerate(CONFIGS):
        folder = Path(directory) / f"task_{task:02d}"
        row = dict.fromkeys(FIELDS, "")
        row.update(task=task, z=z, tau=tau, lambda_anchor=weight, status="missing")
        rows.append(row)
        if not folder.exists():
            row["error"] = "Task directory missing"
            continue
        try:
            status = (folder / "status.txt").read_text().strip()
            if status != "complete":
                row.update(status="failed" if status.startswith("failed") else "incomplete",
                           error=status or "Empty task status")
                continue
            checkpoints = list(folder.glob("attribution_22cls_*.pt"))
            if len(checkpoints) != 1:
                raise ValueError(f"Expected one checkpoint, found {len(checkpoints)}")
            row["checkpoint"] = str(checkpoints[0].resolve())
            checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            expected = dict(lambda_anchor=weight, geodesic_margin=z, target_norm=tau,
                            num_epochs=30, train_augment=True, aug_policy="omnidfa",
                            anchor_init="random", hierarchy="none", lambda_norm=0.5,
                            norm_mode="floor", anchor_penalty="geodesic", anchor_reduction="mean")
            for key, value in expected.items():
                if checkpoint.get(key) != value:
                    raise ValueError(f"Checkpoint {key}: expected {value!r}, got {checkpoint.get(key)!r}")
            epoch = checkpoint["epoch"]
            if not isinstance(epoch, int) or not 1 <= epoch <= 30:
                raise ValueError(f"Invalid selected epoch: {epoch}")
            row.update(epoch=epoch, val_balanced=accuracy(checkpoint["val_balanced"]))
            del checkpoint
            for level, condition in CONDITIONS.items():
                path = folder / "evaluation" / f"test_results_degraded_{level}.txt"
                lines = path.read_text().splitlines()
                for metric in ("acc", "balanced_accuracy"):
                    values = [line.split(":", 1)[1].strip() for line in lines
                              if line.startswith(f"{metric}:")]
                    if len(values) != 1:
                        raise ValueError(f"{path.name}: expected one {metric}")
                    row[f"{condition}_{metric}"] = accuracy(values[0])
            row["status"] = "complete"
        except Exception as exc:
            row.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
    return sorted(rows, key=lambda row: (
        row["status"] != "complete", -float(row["val_balanced"] or -1), row["task"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="One sweep submission's directory")
    args = parser.parse_args(argv)
    if not args.dir.is_dir():
        parser.error(f"Sweep directory does not exist: {args.dir}")
    rows = collect(args.dir)
    output = args.dir / "summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("Test columns: overall / balanced accuracy (%)")
    print(f"{'task':>4} {'z':>2} {'tau':>3} {'weight':>6} {'epoch':>5} {'val_bal':>8} "
          f"{'clean':>15} {'JPEG30':>15} {'BLUR5':>15}  status")
    for row in rows:
        def percent(key):
            return f"{100 * row[key]:.2f}" if row[key] != "" else "-"

        tests = [f"{percent(f'{name}_acc')} / {percent(f'{name}_balanced_accuracy')}"
                 for name in CONDITIONS.values()]
        print(f"{row['task']:>4} {row['z']:>2} {row['tau']:>3} {row['lambda_anchor']:>6} "
              f"{str(row['epoch']):>5} {percent('val_balanced'):>8} "
              + " ".join(f"{value:>15}" for value in tests) + f"  {row['status']}")
        print(f"     checkpoint: {row['checkpoint'] or '-'}")
        if row["error"]:
            print(f"     {row['error']}")
    incomplete = sum(row["status"] != "complete" for row in rows)
    if incomplete:
        print(f"Incomplete sweep: {incomplete}/18 runs missing, failed or invalid; no winner declared.")
    else:
        print(f"Best clean validation configuration: task {rows[0]['task']} "
              f"(z={rows[0]['z']}, tau={rows[0]['tau']}, lambda_anchor={rows[0]['lambda_anchor']})")
    print(f"CSV: {output.resolve()}")
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
