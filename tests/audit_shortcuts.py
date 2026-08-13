"""Can the 22 classes be told apart from FILE METADATA alone?

We get 0.993 clean accuracy from a two-term hinge loss. Before that goes in a paper,
rule out the boring explanation: that the benchmark leaks the label outside the
pixels. One leak is already known and documented — dataset.py:161 accepts .jpg/.jpeg
ONLY for `real`, every synthetic is .png, so "has JPEG artifacts ⇒ real" is free on
clean data — but nobody has ever measured how much the rest of the bookkeeping
(native resolution, aspect ratio, file size, PNG bit depth) gives away.

No pixels are decoded: PIL opens the header lazily and we read the PNG IHDR bytes
directly. A few thousand stat+header reads, no GPU, runs on the login node.

    IAB_EXCLUDE_GENERATORS=dalle3 python -m tests.audit_shortcuts \\
        --root_dir $FAST/datasets/iab_dataset --per_class 500

Read the SECOND accuracy line (geometry only, format features dropped): the first
one is high by construction because of the .jpg leak. If geometry-only is near
chance (4.5%), the shortcut is confined to real-vs-fake and the 22-way result is
about pixels. If it is 0.5+, a slice of our accuracy is bookkeeping.
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

# Format features: the known leak. Reported separately so the interesting number
# (everything else) is not drowned by it.
FORMAT_FEATS = ["is_png", "is_jpeg", "png_bit_depth", "png_color_type", "png_interlace"]
GEOM_FEATS = ["width", "height", "aspect", "area", "filesize", "bytes_per_pixel"]
FEATS = GEOM_FEATS + FORMAT_FEATS


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
        "is_png": int(ext == ".png"), "is_jpeg": int(ext in (".jpg", ".jpeg")),
        "png_bit_depth": bd, "png_color_type": ct, "png_interlace": il,
    }


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

    for name, feats in (("ALL metadata", FEATS), ("GEOMETRY only", GEOM_FEATS)):
        fine, det, imp = tree_accuracy(rows, labels, feats, args.max_depth, args.seed)
        print(f"\n--- decision tree (depth {args.max_depth}) on {name} ---")
        print(f"  22-way balanced accuracy : {fine:.3f}   ({fine / chance:.1f}x chance)")
        print(f"  real-vs-fake balanced acc: {det:.3f}")
        print("  top features: " + ", ".join(f"{f}={v:.2f}" for f, v in imp[:5] if v > 0))

    print("\nOur model is at 0.993 (22-way, clean). Compare the GEOMETRY-only line:\n"
          "that is the part of the label that leaks with no pixels involved and no\n"
          "known excuse.")


if __name__ == "__main__":
    main()
