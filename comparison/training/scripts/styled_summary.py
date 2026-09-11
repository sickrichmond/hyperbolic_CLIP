"""Summarize saved styled-dataset metrics without running models.

Read <root>/<model>/<dataset>/metrics.json for the configured style names.
Report full-label and restricted four-class accuracy, per-class recall,
separate mean recalls for FLUX/SDXL and SD3/SD3_5, off-target predictions,
and family-correct recall from the full-label confusion matrix.

Family groups come from --tree or the HiFi level-3 mapping. Compare in-family
errors with the family-size uniform-error reference. These are descriptive
aggregations, not evidence that style or model lineage caused the errors.

Usage: python -m comparison.training.scripts.styled_summary ROOT --help
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


def family_accuracy(blob, tree):
    """(macro family accuracy, share of errors inside the family, uniform-error null).

    The in-family share means nothing on its own: with the SD family holding 5 of 22
    classes, 4 of a wrong image's 21 alternatives are already in-family, so ~19% comes
    free. The null is that number, weighted by where the errors actually are, and the
    claim is the ratio to it — same logic as the uniform-error null in
    detection_from_confmat.resolution_analysis.

    Read off the 22-way matrix, so predictions landing on the other 18 classes are
    scored honestly: an SD3 image called CogView3_PLUS is cross-family and counts as
    a real error, not as a near miss.
    """
    cm = blob["metrics"]["full_22_way"]["confusion_matrix"]
    labels, matrix = cm["labels"], cm["matrix"]
    fam = [tree.get(l) for l in labels]
    per_class, same, cross = [], 0, 0
    for i, row in enumerate(matrix):
        total = sum(row)
        if not total:
            continue
        hit = sum(n for j, n in enumerate(row) if fam[j] == fam[i])
        per_class.append(hit / total)
        for j, n in enumerate(row):
            if i == j:
                continue
            if fam[j] == fam[i]:
                same += n
            else:
                cross += n
    if not per_class:
        return None, None, None
    k = len(labels)
    sizes = {f: fam.count(f) for f in set(fam)}
    expected = sum((sum(row) - row[i]) * (sizes[fam[i]] - 1) / (k - 1)
                   for i, row in enumerate(matrix) if sum(row))
    errors = same + cross
    return (100 * sum(per_class) / len(per_class),
            100 * same / errors if errors else None,
            100 * expected / errors if errors else None)


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
    p.add_argument("--tree", metavar="JSON",
                   help="{class: family} from extract_tree; default is _HIFI_HIERARCHY level 3")
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

    if args.tree:
        tree = json.loads(Path(args.tree).read_text())
        which = args.tree
    else:
        from comparison.dataset.ImageAttributionDataset.dataset import _HIFI_HIERARCHY
        tree = {k: v[2] for k, v in _HIFI_HIERARCHY.items()}
        which = "_HIFI_HIERARCHY level 3"
    table(f"FAMILY-level accuracy ({which}) — right family, leaf may be wrong", models,
          lambda b: family_accuracy(b, tree)[0],
          "Compare against the 22-way table. A large gap is graceful degradation the model\n"
          "already performs but cannot report: it keeps the family and guesses the leaf.")
    table("share of errors that stay INSIDE the family (%)", models,
          lambda b: family_accuracy(b, tree)[1])
    table("...over the uniform-error null (x)", models,
          lambda b: (lambda f: f[1] / f[2] if f[2] else None)(family_accuracy(b, tree)),
          "The null is what you get for free from family sizes alone (~5x/21 for an SD\n"
          "image), so 1.0x is chance. High = the mistakes are near misses a hierarchy could\n"
          "catch. At 1.0 the representation loses the family too and a hierarchy cannot help.")

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
