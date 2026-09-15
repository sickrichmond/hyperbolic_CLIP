"""
Aggregate the per-level test results of the 4 comparison methods into readable
tables + a CSV. Pure stdlib (no torch/sklearn) — run on a CINECA login node:

    cd $WORK/hyp_fine_tuning/hyperbolic_CLIP
    python comparison/training/scripts/aggregate_test_results.py
    # optional: --split semantic_split_1 , --csv out.csv , --logs_test <dir>

For each method it auto-picks the most recent test_* run and parses
test_results_degraded_{0..6}.txt.
"""
import os
import re
import glob
import argparse

METHODS = ["resnet50", "dct", "hifi_net", "defl"]
# Scalar metrics written by calculate_metrics_for_test (one "key: value" per line).
SCALARS = ["acc", "auc", "ap",
           "precision_macro", "recall_macro", "f1_macro",
           "precision_weighted", "recall_weighted", "f1_weighted"]
# Headline metrics to print as method x level tables.
HEADLINE = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc"]
LEVELS = list(range(7))
LEVEL_LABEL = {0: "clean", 1: "DS0.5", 2: "DS0.25", 3: "JPEG65",
               4: "JPEG30", 5: "Blur3", 6: "Blur5"}

_SCALAR_RE = re.compile(r'^\s*([a-z0-9_]+):\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$')


def parse_file(path):
    """Return {metric: float} for the scalar 'key: value' lines in a result file."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            m = _SCALAR_RE.match(line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def latest_test_dir(logs_test, split, method):
    base = os.path.join(logs_test, split, method)
    dirs = sorted(glob.glob(os.path.join(base, "test_*")), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_test", default="comparison/training/logs_test")
    p.add_argument("--split", default="default_split",
                   help="default_split or semantic_split_{1,2,3}")
    p.add_argument("--csv", default=None, help="optional path to write a full CSV")
    args = p.parse_args()

    # data[method][level] = {metric: value}
    data = {}
    for method in METHODS:
        tdir = latest_test_dir(args.logs_test, args.split, method)
        if tdir is None:
            print(f"[warn] no test run found for {method} under {args.split}")
            data[method] = {}
            continue
        print(f"{method:10s} <- {tdir}")
        data[method] = {
            lvl: parse_file(os.path.join(tdir, f"test_results_degraded_{lvl}.txt"))
            for lvl in LEVELS
        }
    print()

    # Headline tables: one per metric, rows = methods, cols = levels.
    for metric in HEADLINE:
        header = "  ".join(f"{LEVEL_LABEL[l]:>8s}" for l in LEVELS)
        print(f"=== {metric} ===")
        print(f"{'method':10s}  {header}")
        for method in METHODS:
            cells = []
            for lvl in LEVELS:
                v = data[method].get(lvl, {}).get(metric)
                cells.append(f"{v:8.4f}" if v is not None else f"{'-':>8s}")
            print(f"{method:10s}  " + "  ".join(cells))
        print()

    # Full CSV: method,level,label,<all scalars>
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("method,level,degradation," + ",".join(SCALARS) + "\n")
            for method in METHODS:
                for lvl in LEVELS:
                    row = data[method].get(lvl, {})
                    vals = ",".join(
                        f"{row[s]:.6f}" if s in row else "" for s in SCALARS)
                    f.write(f"{method},{lvl},{LEVEL_LABEL[lvl]},{vals}\n")
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
