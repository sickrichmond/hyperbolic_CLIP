"""
Parse the baselines' per-level test metrics straight from their SLURM .out logs
(each job streams `Test metrics: {...}` to stdout via the logger's StreamHandler),
and print a method x level table + optional CSV. Pure stdlib.

Use when the test_results_degraded_*.txt files aren't reachable but the job .out
logs are. Filters by job id so you don't mix pre-fix (old protocol) and post-fix
(leakage-free / same-images) runs.

Run from wherever the iab_*_test_*.out files live, e.g.:
    cd $WORK/hyp_fine_tuning/hyperbolic_CLIP
    python $WORK/hyp_fine_tuning/hyperbolic_CLIP_riccardo/comparison/training/scripts/parse_baseline_out.py \
        --min_jobid 49000000 --csv $WORK/outputs/baseline_results.csv
"""
import os
import re
import csv
import glob
import argparse

LVL = {0: "clean", 1: "DS0.5", 2: "DS0.25", 3: "JPEG65",
       4: "JPEG30", 5: "Blur3", 6: "Blur5"}
NAME = {"rn50": "resnet50", "dct": "dct", "hifi": "hifi_net", "defl": "defl"}
SCAL = ["acc", "auc", "ap", "precision_macro", "recall_macro", "f1_macro"]

_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
LVL_RE = re.compile(r"Testing with degraded level (\d+)")
SC_RE = re.compile(r"'(" + "|".join(SCAL) + r")':\s*" + _NUM)
FN_RE = re.compile(r"iab_([a-z0-9]+)_test_(\d+)\.out$")


def parse_out(path):
    """Yield (level, {metric: value}) for each level's `Test metrics:` line."""
    cur = None
    for line in open(path, errors="ignore"):
        g = LVL_RE.search(line)
        if g:
            cur = int(g.group(1))
        if "Test metrics:" in line and cur is not None:
            row = {}
            for k, v in SC_RE.findall(line):
                row[k] = float(v)
            if row:
                yield cur, row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=".", help="dir holding iab_*_test_*.out")
    p.add_argument("--min_jobid", type=int, default=49000000,
                   help="only include runs with SLURM job id >= this (post-fix runs)")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    data = {}   # data[method][level] = {metric: val}
    src = {}    # src[method][level]  = jobid (provenance)
    files = []
    for f in glob.glob(os.path.join(args.dir, "iab_*_test_*.out")):
        m = FN_RE.search(os.path.basename(f))
        if not m:
            continue
        method = NAME.get(m.group(1))
        jobid = int(m.group(2))
        if method and jobid >= args.min_jobid:
            files.append((jobid, method, f))

    print(f"Files included (job id >= {args.min_jobid}):")
    for jobid, method, f in sorted(files):
        seen = []
        for lvl, row in parse_out(f):
            data.setdefault(method, {})[lvl] = row
            src.setdefault(method, {})[lvl] = jobid
            seen.append(lvl)
        print(f"  {method:10s} job {jobid}  levels {sorted(set(seen))}")

    for metric in SCAL:
        print(f"\n=== {metric} ===")
        print(f"{'method':10s}  " + "  ".join(f"{LVL[l]:>8s}" for l in range(7)))
        for method in ["resnet50", "dct", "hifi_net", "defl"]:
            cells = []
            for l in range(7):
                v = data.get(method, {}).get(l, {}).get(metric)
                cells.append(f"{v:8.4f}" if v is not None else f"{'-':>8s}")
            print(f"{method:10s}  " + "  ".join(cells))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "level", "degradation", *SCAL, "src_jobid"])
            for method in ["resnet50", "dct", "hifi_net", "defl"]:
                for l in range(7):
                    r = data.get(method, {}).get(l, {})
                    w.writerow([method, l, LVL[l],
                                *[r.get(s, "") for s in SCAL],
                                src.get(method, {}).get(l, "")])
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
