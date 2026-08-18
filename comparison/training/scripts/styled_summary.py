"""The styled comparison, read out of the metrics.json the harness already wrote.

The headline table (HiFi-Net 99.19 ... HypCLIP 83.31) invites one reading: CLIP-based
methods lose ~16 points under a content shift. But two of the four target generators
are SD3 and SD3_5 — the pair every model confuses on clean IAB too — so part of that
gap is pre-existing twin confusion and not style at all. Splitting the two is the
whole point of this script, and it needs no GPU: per_class.recall is in every file.

Reports, per model and style:
  * 22-way and restricted 4-way accuracy (the published columns)
  * per-class recall for the four targets
  * TWIN-FREE accuracy: the mean over FLUX and SDXL alone, next to the mean over
    SD3 and SD3_5. If the twin-free number is flat across styles, the styled gap is a
    twin problem wearing a style costume.
  * where the off-target mass goes — under JPEG everything drifts to `real`, and this
    says whether a semantic shift does the same or something else.

    python -m comparison.training.scripts.styled_summary $R
    python -m comparison.training.scripts.styled_summary $R --model hypclip --verbose

Pure stdlib, login node. $R is the directory holding <model>/<dataset>/metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

STYLES = {"iab_recap_dataset_v2": "standard", "iab_recap_cartoon_v2": "cartoon",
          "iab_recap_clipart_v2": "clipart", "iab_recap_photorealistic_v2": "photoreal"}
ORDER = ["standard", "cartoon", "clipart", "photoreal"]
TWINS = ("SD3", "SD3_5")
SOLO = ("FLUX", "SDXL")


def load(root):
    """{model: {style: metrics.json}} — missing cells are simply absent."""
    out = {}
    for model_dir in sorted(Path(root).iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in model_dir.iterdir():
            f = ds_dir / "metrics.json"
            if f.is_file() and ds_dir.name in STYLES:
                out.setdefault(model_dir.name, {})[STYLES[ds_dir.name]] = json.loads(f.read_text())
    return out


def recalls(blob, view="full_22_way"):
    per = blob["metrics"][view]["per_class"]
    return {c: v["recall"] for c, v in per.items() if v["support"]}


def table(title, models, cell, note=""):
    print(f"\n### {title}\n")
    print("| model | " + " | ".join(ORDER) + " | mean |")
    print("|---|" + "|".join([":--:"] * (len(ORDER) + 1)) + "|")
    for m, styles in models.items():
        vals = [cell(styles[s]) if s in styles else None for s in ORDER]
        got = [v for v in vals if v is not None]
        cells = [f"{v:.2f}" if v is not None else "—" for v in vals]
        mean = f"{sum(got) / len(got):.2f}" if got else "—"
        print(f"| {m} | " + " | ".join(cells) + f" | **{mean}** |")
    if note:
        print(f"\n{note}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="aug_test_results_styled_method_comparison_* directory")
    p.add_argument("--model", help="only this model, with the per-class detail")
    p.add_argument("--verbose", action="store_true", help="per-class recall for every model")
    args = p.parse_args()

    models = load(args.root)
    if args.model:
        models = {k: v for k, v in models.items() if k == args.model}
    if not models:
        sys.exit(f"no metrics.json found under {args.root}")
    print(f"{len(models)} models x {len(ORDER)} styles, from {args.root}")

    table("22-way accuracy (the published column)", models,
          lambda b: 100 * b["metrics"]["full_22_way"]["accuracy"])
    table("restricted 4-way accuracy", models,
          lambda b: 100 * b["metrics"]["restricted_4_way"]["accuracy"])

    def twin_free(b):
        r = recalls(b)
        got = [r[c] for c in SOLO if c in r]
        return 100 * sum(got) / len(got) if got else None

    def twin_only(b):
        r = recalls(b)
        got = [r[c] for c in TWINS if c in r]
        return 100 * sum(got) / len(got) if got else None

    table("TWIN-FREE accuracy — mean recall over FLUX and SDXL only", models, twin_free,
          "Flat across styles ⇒ the styled gap is the twins, not the style.")
    table("the twins alone — mean recall over SD3 and SD3_5", models, twin_only)
    table("off-target rate (%)", models,
          lambda b: 100 * b["metrics"]["full_22_way"]["off_target_prediction_rate"])

    if args.model or args.verbose:
        for m, styles in models.items():
            print(f"\n### {m} — per-class recall, and where the off-target mass goes\n")
            print("| style | " + " | ".join(SOLO + TWINS) + " | off-target destinations (top 4) |")
            print("|---|" + "|".join([":--:"] * 4) + "|---|")
            for s in ORDER:
                if s not in styles:
                    continue
                r = recalls(styles[s])
                off = styles[s]["metrics"]["full_22_way"]["off_target_predictions"]
                top = sorted(off.items(), key=lambda kv: -kv[1])[:4]
                cells = [f"{100 * r[c]:.1f}" if c in r else "—" for c in SOLO + TWINS]
                print(f"| {s} | " + " | ".join(cells) + " | "
                      + ", ".join(f"{k} {v}" for k, v in top) + " |")


if __name__ == "__main__":
    main()
