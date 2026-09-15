"""Measure class predictability from sampled image-file metadata.

Use the harness class/path conventions to sample files, reading image headers
and PNG IHDR fields without decoding pixels. Fit depth-limited decision trees
on metadata subsets with a stratified 70/30 split; report generator and
real-vs-fake balanced accuracy, feature importance and resolution frequencies.

The aspect and scale_to_224 subset describes preprocessing-related covariates.
Metadata predictability identifies possible confounds, not proof that an image
classifier uses them. This is not the harness model-evaluation split.

Usage: python -m tests.audit_shortcuts --help
"""
import argparse
import os
import random
import struct
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from comparison.dataset.ImageAttributionDataset.dataset import (
    model_class_to_label, semantic_to_relpath)

# Report file-format covariates separately from the other metadata features.
FORMAT_FEATS = ["is_png", "is_jpeg", "png_bit_depth", "png_color_type", "png_interlace"]
GEOM_FEATS = ["width", "height", "aspect", "area", "filesize", "bytes_per_pixel"]
FEATS = GEOM_FEATS + FORMAT_FEATS
# Aspect and native-to-224 scale can affect pixels after CLIP preprocessing.
# Metadata predictability does not establish that the image model uses it.
SURVIVING = ["aspect", "scale_to_224"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root_dir", required=True)
    p.add_argument("--per_class", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_depth", type=int, default=6)
    return p.parse_args()


def png_ihdr(path):
    """(bit_depth, color_type, interlace) straight out of the IHDR chunk, or zeros."""
    with open(path, "rb") as f:
        head = f.read(29)
    # 0-7 signature | 8-11 length | 12-15 "IHDR" | 16-23 w,h | 24 depth | 25 colour
    # | 26 compression | 27 filter | 28 interlace
    if len(head) < 29 or head[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0, 0
    _, _, bit_depth, color_type, _, _, interlace = struct.unpack(">II5B", head[16:29])
    return bit_depth, color_type, interlace


def features(path):
    with Image.open(path) as im:          # header only, no decode
        w, h = im.size
    size = os.path.getsize(path)
    ext = path.suffix.lower()
    bd, ct, il = png_ihdr(path) if ext == ".png" else (0, 0, 0)
    return {
        "width": w, "height": h, "aspect": w / max(h, 1), "area": w * h,
        "filesize": size, "bytes_per_pixel": size / max(w * h, 1),
        # CLIPImageProcessor resizes the SHORTEST edge to 224, so this is the exact
        # resampling ratio the pixels go through before the model ever sees them.
        "scale_to_224": 224 / max(min(w, h), 1),
        "is_png": int(ext == ".png"), "is_jpeg": int(ext in (".jpg", ".jpeg")),
        "png_bit_depth": bd, "png_color_type": ct, "png_interlace": il,
    }


def resolution_table(rows, labels):
    """Print native-size frequencies and classes dominated by one resolution."""
    per_cls = defaultdict(Counter)
    for r, y in zip(rows, labels):
        per_cls[y][(r["width"], r["height"])] += 1
    print(f"\n--- native resolutions ---\n{'class':16s} {'#sizes':>7} {'top-1 share':>12}"
          f"   most common")
    for cls in sorted(per_cls):
        c = per_cls[cls]
        n = sum(c.values())
        top = c.most_common(3)
        share = top[0][1] / n
        pretty = ", ".join(f"{w}x{h} ({100 * k / n:.0f}%)" for (w, h), k in top)
        print(f"{cls:16s} {len(c):7d} {share:12.2f}   {pretty}")
    fixed = [c for c in per_cls if per_cls[c].most_common(1)[0][1] / sum(per_cls[c].values()) > 0.95]
    print(f"\n{len(fixed)}/{len(per_cls)} classes emit at essentially ONE resolution: "
          f"{', '.join(sorted(fixed)) or '—'}")


def collect(root, per_class, seed):
    """Same enumeration rule as the harness (dataset.py:_make_dataset), sampled."""
    rng = random.Random(seed)
    rows, labels, exts = [], [], defaultdict(Counter)
    for cls in model_class_to_label:
        paths = []
        for relpath in semantic_to_relpath.values():
            d = Path(root) / cls / relpath
            if not d.is_dir():
                continue
            ok = (".png", ".jpg", ".jpeg") if cls == "real" else (".png",)
            paths += [p for p in d.iterdir() if p.suffix.lower() in ok]
        if not paths:
            print(f"  ⚠️  no files for {cls}")
            continue
        rng.shuffle(paths)
        for p in paths[:per_class]:
            exts[cls][p.suffix.lower()] += 1
            rows.append(features(p))
            labels.append(cls)
        print(f"  {cls:16s} {len(paths[:per_class]):5d} sampled of {len(paths):7d}  "
              f"{dict(exts[cls])}")
    return rows, labels, exts


def tree_accuracy(rows, labels, feats, max_depth, seed):
    """22-way and real-vs-fake balanced accuracy of a depth-limited tree."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X = [[r[f] for f in feats] for r in rows]
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.3,
                                          random_state=seed, stratify=labels)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    fine = balanced_accuracy_score(yte, pred)

    det_true = [y == "real" for y in yte]
    det_pred = [p == "real" for p in pred]
    det = balanced_accuracy_score(det_true, det_pred)
    imp = sorted(zip(feats, clf.feature_importances_), key=lambda kv: -kv[1])
    return fine, det, imp


def main():
    args = parse_args()
    print(f"Enumerating {args.root_dir} ({len(model_class_to_label)} classes, "
          f"{args.per_class}/class)…")
    rows, labels, exts = collect(args.root_dir, args.per_class, args.seed)
    K = len(set(labels))
    chance = 1.0 / K
    print(f"\n{len(rows)} files, {K} classes, chance = {chance:.3f}")

    real_ext = exts.get("real", Counter())
    total_real = sum(real_ext.values()) or 1
    print(f"real formats: {dict(real_ext)}  "
          f"({100 * (real_ext['.jpg'] + real_ext['.jpeg']) / total_real:.1f}% JPEG)")

    resolution_table(rows, labels)

    for name, feats in (("ALL metadata", FEATS),
                        ("GEOMETRY only", GEOM_FEATS),
                        ("SURVIVES the CLIP preprocessing", SURVIVING),
                        ("scale_to_224 ALONE", ["scale_to_224"])):
        fine, det, imp = tree_accuracy(rows, labels, feats, args.max_depth, args.seed)
        print(f"\n--- decision tree (depth {args.max_depth}) on {name} ---")
        print(f"  22-way balanced accuracy : {fine:.3f}   ({fine / chance:.1f}x chance)")
        print(f"  real-vs-fake balanced acc: {det:.3f}")
        print("  top features: " + ", ".join(f"{f}={v:.2f}" for f, v in imp[:5] if v > 0))

    print("\nHow to read this. The model never sees a NUMBER: CLIPImageProcessor resizes\n"
          "the shortest edge to 224 and centre-crops, so area / filesize / bytes_per_pixel\n"
          "are gone by the time it looks. The first two blocks therefore measure how much\n"
          "the BENCHMARK leaks, not how much we cheat.\n"
          "The third and fourth are the ones that implicate the model: aspect survives as\n"
          "crop framing, and scale_to_224 survives as a resampling signature in the pixels.\n"
          "If scale_to_224 alone separates the classes, then resolution is confounded with\n"
          "the label and part of our accuracy may be resampling statistics rather than\n"
          "generator fingerprints. The control is a common-resolution eval — see the\n"
          "`--pre_resize` note in comparison/training/test_hypclip.py.")


if __name__ == "__main__":
    main()
