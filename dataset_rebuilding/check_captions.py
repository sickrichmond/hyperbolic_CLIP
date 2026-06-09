"""
Quality-check (and optionally clean) the re-generated caption CSVs.

Does two jobs:

1. **Dedup / clean** — the two overlapping SLURM jobs wrote to the same CSVs, so
   each image may appear more than once and a stray header line ("ImgPath,...")
   may be interleaved. This drops duplicate stems (keeping the last caption) and
   removes stray headers. With --write_clean it writes the cleaned CSVs out
   (non-destructive: to a separate directory).

2. **Validate** — per semantic class it reports:
     - rows raw vs unique stems (how many duplicates were removed)
     - coverage vs the real images on disk (should be 100% / 2000 per class)
     - empty/failed captions
     - caption word-count stats and the % inside the 40-80 word target.

Read-only by default. Example:

    python dataset_rebuilding/check_captions.py \\
        --captions_dir $WORK/iab_captions_detailed \\
        --dataset_path $WORK/iab_dataset \\
        --samples 3 \\
        --write_clean $WORK/iab_captions_detailed_clean
"""
import argparse
import csv
import random
import statistics as st
import sys
from pathlib import Path

# Mirrors caption_real_images.py (kept local so QC has no heavy deps).
SEMANTIC_DIR = {
    "COCO":        (None,         "COCO"),
    "ImageNet-1k": (None,         "ImageNet-1k"),
    "cat":         ("AnimalFace", "cat"),
    "dog":         ("AnimalFace", "dog"),
    "wild":        ("AnimalFace", "wild"),
    "FFHQ":        ("HumanFace",  "FFHQ"),
    "celebahq":    ("HumanFace",  "celebahq"),
    "bedroom":     ("Scene",      "bedroom"),
    "church":      ("Scene",      "church"),
    "classroom":   ("Scene",      "classroom"),
}
OUTPUT_CSV = {
    "COCO":        "COCO.csv",
    "cat":         "AnimalFace_cat.csv",
    "dog":         "AnimalFace_dog.csv",
    "wild":        "AnimalFace_wild.csv",
    "FFHQ":        "HumanFace_FFHQ.csv",
    "celebahq":    "HumanFace_celebahq.csv",
    "bedroom":     "Scene_LSUN_bedroom.csv",
    "church":      "Scene_LSUN_church.csv",
    "classroom":   "Scene_LSUN_classroom.csv",
    "ImageNet-1k": "imagenet-1k.csv",
}
THREE_COLUMN = {"ImageNet-1k"}
ALL_SEMANTICS = list(SEMANTIC_DIR.keys())
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPEG"}
WORD_LO, WORD_HI = 40, 80


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captions_dir", required=True)
    p.add_argument("--dataset_path", default=None,
                   help="IAB root — enables coverage check vs real/ images.")
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    p.add_argument("--samples", type=int, default=0,
                   help="Print this many random captions per class.")
    p.add_argument("--write_clean", default=None,
                   help="Write deduped/cleaned CSVs to this directory.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _img_dir(root: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (root / "real" / leaf) if sup is None else (root / "real" / sup / leaf)


def _n_images(root: Path, semantic: str) -> int:
    d = _img_dir(root, semantic)
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix in IMAGE_EXTS)


def load_clean(csv_path: Path):
    """Return (rows_by_stem, n_raw_valid, n_dupes, n_header_junk).
    rows_by_stem keeps the LAST occurrence per stem; stray header lines dropped.
    """
    rows_by_stem: dict[str, list[str]] = {}
    n_raw = n_dupes = n_junk = 0
    if not csv_path.exists():
        return rows_by_stem, 0, 0, 0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "ImgPath" or Path(row[0]).stem == "ImgPath":
                n_junk += 1                      # header (possibly repeated)
                continue
            n_raw += 1
            stem = Path(row[0]).stem
            if stem in rows_by_stem:
                n_dupes += 1
            rows_by_stem[stem] = row
    return rows_by_stem, n_raw, n_dupes, n_junk


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    caps_dir = Path(args.captions_dir)
    root = Path(args.dataset_path) if args.dataset_path else None
    clean_dir = Path(args.write_clean) if args.write_clean else None
    if clean_dir:
        clean_dir.mkdir(parents=True, exist_ok=True)

    print(f"Captions dir: {caps_dir}")
    print(f"{'class':12s} {'uniq':>6s} {'dupes':>6s} {'cover':>7s} "
          f"{'empty':>6s} {'words(min/med/max)':>20s} {'%40-80':>7s}")
    print("-" * 78)

    tot_uniq = tot_dupes = tot_empty = tot_imgs = 0
    all_in_band = 0
    for sem in args.semantics:
        csv_path = caps_dir / OUTPUT_CSV[sem]
        rows_by_stem, n_raw, n_dupes, n_junk = load_clean(csv_path)
        three = sem in THREE_COLUMN
        caps = [r[-1].strip() for r in rows_by_stem.values()]
        empties = sum(1 for c in caps if not c)
        wcs = [len(c.split()) for c in caps if c]
        n_imgs = _n_images(root, sem) if root else 0
        cover = (f"{100*len(rows_by_stem)/n_imgs:5.1f}%" if n_imgs else "   n/a")
        in_band = sum(1 for w in wcs if WORD_LO <= w <= WORD_HI)
        pct_band = f"{100*in_band/len(wcs):5.1f}%" if wcs else "  n/a"
        wstat = (f"{min(wcs):3d}/{int(st.median(wcs)):3d}/{max(wcs):3d}"
                 if wcs else "n/a")
        print(f"{sem:12s} {len(rows_by_stem):6d} {n_dupes:6d} {cover:>7s} "
              f"{empties:6d} {wstat:>20s} {pct_band:>7s}"
              + (f"  [+{n_junk} hdr]" if n_junk > 1 else ""))

        tot_uniq += len(rows_by_stem); tot_dupes += n_dupes
        tot_empty += empties; tot_imgs += n_imgs; all_in_band += in_band

        if args.samples and caps:
            for c in rng.sample(caps, min(args.samples, len(caps))):
                print(f"    · ({len(c.split())}w) {c[:160]}{'…' if len(c) > 160 else ''}")

        if clean_dir:
            out = clean_dir / OUTPUT_CSV[sem]
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ImgPath", "Label", "Caption"] if three
                           else ["ImgPath", "Caption"])
                for stem in sorted(rows_by_stem):
                    w.writerow(rows_by_stem[stem])

    print("-" * 78)
    cover_tot = f"{100*tot_uniq/tot_imgs:5.1f}%" if tot_imgs else "n/a"
    print(f"{'TOTAL':12s} {tot_uniq:6d} {tot_dupes:6d} {cover_tot:>7s} "
          f"{tot_empty:6d}  unique captions; "
          f"{100*all_in_band/max(tot_uniq-tot_empty,1):.1f}% within {WORD_LO}-{WORD_HI} words")
    if clean_dir:
        print(f"\nCleaned CSVs written to: {clean_dir}")
    if tot_empty:
        print(f"WARNING: {tot_empty} empty captions — re-run the captioner to fill them.")


if __name__ == "__main__":
    main()
