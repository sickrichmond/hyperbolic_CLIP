"""Real-vs-fake detection, recovered from confusion matrices already on disk.

The 22x22 conf_matrix is written into every test_results_degraded_*.txt and read
by nothing: the existing aggregators parse only the nine scalars. Everything needed
for a detection number is therefore already there, at zero GPU cost.

Plain accuracy on this collapse is meaningless — the test set is ~95.5% synthetic,
so never predicting `real` scores ~0.955. Only balanced accuracy is reported.

Also cross-tabs accuracy against the native-resolution groups (see RES_GROUP), which
is how we tell whether the detection number is a fingerprint or a resampling
artifact — tests/audit_shortcuts.py showed 20 of the 22 classes emit at exactly one
resolution and that scale_to_224 alone separates real from fake at 0.866.

No GPU, login node is fine. Run from the repo root:

    IAB_EXCLUDE_GENERATORS=dalle3 python -m comparison.training.scripts.detection_from_confmat \\
        sweepwin=$WORK/outputs/hypclip_fair_22cls \\
        resnet50=comparison/training/logs/default_split/resnet50/<run>
"""
import re
import sys
from pathlib import Path

LEVELS = {0: "clean", 1: "DS0.5", 2: "DS0.25", 3: "JPEG65",
          4: "JPEG30", 5: "Blur3", 6: "Blur5"}

# Native shortest edge per class, measured by tests/audit_shortcuts.py: 20 of the 22
# classes emit at exactly ONE resolution. CLIPImageProcessor scales the shortest edge
# to 224, so classes sharing a value go through an IDENTICAL resampling ratio and the
# resolution channel cannot separate them. 14 of the 22 sit in the 1024 group.
RES_GROUP = {
    "janus-pro": 384,
    "FLUX": 512, "SD1_5": 512,
    "KANDINSKY": 768, "SD2_1": 768, "grok3": 768,
    "4o": 1024, "CogView3_PLUS": 1024, "PIXART": 1024, "PLAYGROUND_2_5": 1024,
    "SD3": 1024, "SD3_5": 1024, "SDXL": 1024, "hidream": 1024, "hunyuan": 1024,
    "ideogram": 1024, "infinity": 1024, "kling": 1024, "mid-5.2": 1024,
    "mid-6.0": 1024,
    "gemini": 0, "real": 0,          # 0 = many native sizes, no fixed ratio
}


def parse_conf(path):
    """The KxK matrix, read by BRACKET NESTING rather than "everything after the key".

    test_hypclip.py happens to write conf_matrix last, but the baselines iterate
    test_metrics.items() (test.py:114-118), so semantic_acc and the *_per_class lists
    can follow it — and their digits would be swallowed by a greedier parser.
    """
    txt = path.read_text()
    head = txt.find("conf_matrix:")
    if head < 0:
        return None
    start = txt.find("[", head)
    if start < 0:
        return None
    depth, end = 0, None
    for i in range(start, len(txt)):
        if txt[i] == "[":
            depth += 1
        elif txt[i] == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"{path}: unterminated conf_matrix")
    nums = [int(n) for n in re.findall(r"-?\d+", txt[start:end])]
    k = int(round(len(nums) ** 0.5))
    if k * k != len(nums):
        raise ValueError(f"{path}: {len(nums)} ints is not a square matrix")
    return [nums[i * k:(i + 1) * k] for i in range(k)]


def detection(conf, real):
    """(real_recall, fake_recall, balanced, pred_real_rate) from a KxK matrix."""
    k = len(conf)
    tot_real = sum(conf[real]) or 1
    real_rec = conf[real][real] / tot_real
    fake_hits = sum(conf[i][j] for i in range(k) if i != real
                    for j in range(k) if j != real)
    tot_fake = sum(sum(conf[i]) for i in range(k) if i != real) or 1
    fake_rec = fake_hits / tot_fake
    pred_real = sum(conf[i][real] for i in range(k)) / max(
        sum(sum(r) for r in conf), 1)
    return real_rec, fake_rec, (real_rec + fake_rec) / 2, pred_real


def resolution_analysis(conf, names):
    """Is the accuracy explained by the resampling ratio, or not?

    Two numbers, both from the clean confusion matrix:
      - recall INSIDE the 1024 group (14 classes that share one ratio, so resolution
        gives nothing there). High recall = the model has real fingerprints.
      - what fraction of errors stay INSIDE a resolution group. A model reading the
        ratio literally cannot confuse across groups, so ~1.0 is the signature of a
        resolution shortcut — though twin pairs inflate it too, since twins happen to
        share a resolution. It is decisive only in the other direction: well below 1
        means the ratio is NOT the channel.
    """
    k = len(conf)
    grp = [RES_GROUP.get(n, 0) for n in names]
    print("\n### resolution groups (clean)\n")
    print("| group (shortest edge) | classes | mean recall |")
    print("|---|:--:|:--:|")
    for g in sorted({x for x in grp}):
        idx = [i for i in range(k) if grp[i] == g]
        recs = [conf[i][i] / (sum(conf[i]) or 1) for i in idx]
        label = "variable" if g == 0 else f"{g}px  (ratio {224 / g:.3f})"
        print(f"| {label} | {len(idx)} | {sum(recs) / len(recs):.4f} |")

    inside = outside = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            # Only pairs where both classes have a FIXED resolution can be said to
            # be inside or outside a group at all.
            if grp[i] == 0 or grp[j] == 0:
                continue
            if grp[i] == grp[j]:
                inside += conf[i][j]
            else:
                outside += conf[i][j]
    tot = inside + outside or 1
    print(f"\nerrors between fixed-resolution classes: {inside / tot:.3f} inside the same "
          f"group, {outside / tot:.3f} across groups  ({tot} errors)")
    print("Across-group errors are ones the resolution channel would have prevented.")


def routing(conf, real, family):
    """Where the errors on SYNTHETIC images go: ->real / in-family / cross-family."""
    k = len(conf)
    to_real = same = cross = 0
    for i in range(k):
        if i == real:
            continue
        for j in range(k):
            if i == j:
                continue
            n = conf[i][j]
            if j == real:
                to_real += n
            elif family[i] == family[j]:
                same += n
            else:
                cross += n
    tot = to_real + same + cross or 1
    return to_real / tot, same / tot, cross / tot


def main(dirs):
    from comparison.dataset.ImageAttributionDataset.dataset import (
        _HIFI_HIERARCHY, model_class_to_label)

    idx_to_name = {v: k for k, v in model_class_to_label.items()}
    names = [idx_to_name[i] for i in range(len(idx_to_name))]
    real = names.index("real")
    family = [_HIFI_HIERARCHY[n][2] for n in names]
    print(f"{len(names)} classes, real = index {real}\n")

    rows, clean, skipped = {}, {}, {}
    for spec in dirs:
        label, _, path = spec.partition("=")
        d = Path(path or label)
        for f in sorted(d.glob("test_results_degraded_*.txt")):
            lvl = int(re.search(r"_(\d+)\.txt$", f.name).group(1))
            conf = parse_conf(f)
            if conf is None:
                continue
            if len(conf) != len(names):
                # 23-class runs (hypclip_fair) and diagnostics live in the same tree.
                skipped.setdefault(label, len(conf))
                continue
            rows[(label, lvl)] = (detection(conf, real), routing(conf, real, family))
            if lvl == 0:
                clean[label] = conf

    if skipped:
        print("skipped (class count does not match this label map): "
              + ", ".join(f"{m} ({k} classes)" for m, k in sorted(skipped.items())))
    methods = sorted({m for m, _ in rows})
    if not methods:
        sys.exit("no usable results found")
    for title, pick in (("balanced detection accuracy (real vs fake)", lambda d: d[0][2]),
                        ("recall(real)", lambda d: d[0][0]),
                        ("recall(fake)", lambda d: d[0][1]),
                        ("fraction predicted `real`", lambda d: d[0][3])):
        print(f"\n### {title}\n")
        print("| degradation | " + " | ".join(methods) + " |")
        print("|---|" + "|".join([":--:"] * len(methods)) + "|")
        for lvl, name in LEVELS.items():
            if not any((m, lvl) in rows for m in methods):
                continue
            cells = [f"{pick(rows[(m, lvl)]):.3f}" if (m, lvl) in rows else "—"
                     for m in methods]
            print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n### error routing on synthetic images (→real / in-family / cross-family)\n")
    print("| degradation | " + " | ".join(methods) + " |")
    print("|---|" + "|".join([":--:"] * len(methods)) + "|")
    for lvl, name in LEVELS.items():
        if not any((m, lvl) in rows for m in methods):
            continue
        cells = ["{:.2f}/{:.2f}/{:.2f}".format(*rows[(m, lvl)][1])
                 if (m, lvl) in rows else "—" for m in methods]
        print(f"| {name} | " + " | ".join(cells) + " |")

    for label, conf in clean.items():
        print(f"\n{'=' * 60}\n  {label}")
        resolution_analysis(conf, names)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: detection_from_confmat.py label=DIR [label=DIR ...]")
    main(sys.argv[1:])
