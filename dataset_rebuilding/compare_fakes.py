"""
Build side-by-side contact sheets to eyeball the regenerated fakes:

    REAL  |  original IAB fake  |  new fake (from the dense caption)

One montage PNG per (generator, semantic). Lets you judge realism and how the
richer prompt changed the fake, by copying just a handful of images instead of
hundreds.

Pairing
-------
- new fake:      <recap_root>/<GEN>/<semantic>/<stem>.png   (named by real stem)
- real:          <dataset_path>/real/<semantic>/<stem>.<ext>
- original fake: <dataset_path>/<GEN>/<semantic>/<prefix>_p{N}_i{K}.png
                 where N = the row index of <stem> in the fake-source caption CSV
                 (sequential for most classes; looked up by stem for ImageNet).

Example (on CINECA):
    python dataset_rebuilding/compare_fakes.py \\
        --dataset_path      $WORK/iab_dataset \\
        --recap_root        $WORK/iab_recap_dataset \\
        --orig_captions_dir $WORK/hyp_fine_tuning/iab_captions \\
        --generator FLUX --num_samples 8 \\
        --out_dir $WORK/recap_compare
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# IAB layout.
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
# semantic → (fake filename prefix, fake-source CSV) used to map stem → row N.
FAKE_SRC = {
    "COCO":        ("COCO-new",             "COCO-new.csv"),
    "cat":         ("AnimalFace_cat",        "AnimalFace_cat.csv"),
    "dog":         ("AnimalFace_dog",        "AnimalFace_dog.csv"),
    "wild":        ("AnimalFace_wild",       "AnimalFace_wild.csv"),
    "FFHQ":        ("HumanFace_FFHQ",        "HumanFace_FFHQ.csv"),
    "celebahq":    ("HumanFace_celebahq",    "HumanFace_celebahq.csv"),
    "bedroom":     ("Scene_LSUN_bedroom",    "Scene_LSUN_bedroom.csv"),
    "church":      ("Scene_LSUN_church",     "Scene_LSUN_church.csv"),
    "classroom":   ("Scene_LSUN_classroom",  "Scene_LSUN_classroom.csv"),
    "ImageNet-1k": ("imagenet-1k-new",       "imagenet-1k-new.csv"),
}
ALL_SEMANTICS = list(SEMANTIC_DIR.keys())
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPEG")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True, help="IAB root (real/ + <GEN>/).")
    p.add_argument("--recap_root", required=True, help="New fakes root (<GEN>/...).")
    p.add_argument("--orig_captions_dir", required=True,
                   help="Original IAB caption CSVs (for stem→row mapping).")
    p.add_argument("--generator", default="FLUX")
    p.add_argument("--semantics", nargs="+", default=ALL_SEMANTICS,
                   choices=ALL_SEMANTICS)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--thumb", type=int, default=320, help="Thumbnail px per cell.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--allow_missing_new", action="store_true",
                   help="Include samples whose new fake is missing (for local testing).")
    return p.parse_args()


def _sub(parent: Path, semantic: str) -> Path:
    sup, leaf = SEMANTIC_DIR[semantic]
    return (parent / leaf) if sup is None else (parent / sup / leaf)


def _real_dir(root: Path, s):  return _sub(root / "real", s)
def _orig_dir(root, gen, s):   return _sub(root / gen, s)
def _new_dir(recap, gen, s):   return _sub(recap / gen, s)


def _stem_to_row(orig_caps: Path, csv_name: str) -> dict[str, int]:
    path = orig_caps / csv_name
    out: dict[str, int] = {}
    if not path.exists():
        return out
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)                       # header
        for i, row in enumerate(reader):
            if row:
                out[Path(row[0]).stem] = i
    return out


def _orig_fake_path(orig_dir: Path, prefix: str, n: int) -> Path | None:
    for k in (0, 1, 2, 3):
        cand = orig_dir / f"{prefix}_p{n}_i{k}.png"
        if cand.exists():
            return cand
    return None


def _real_path(real_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        cand = real_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _load_thumb(path: Path | None, size: int):
    if path is None or not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    return img


def build_sheet(args, semantic: str) -> bool:
    root = Path(args.dataset_path)
    recap = Path(args.recap_root)
    gen = args.generator
    prefix, csv_name = FAKE_SRC[semantic]
    stem2row = _stem_to_row(Path(args.orig_captions_dir), csv_name)

    real_dir = _real_dir(root, semantic)
    orig_dir = _orig_dir(root, gen, semantic)
    new_dir = _new_dir(recap, gen, semantic)

    new_stems = ({p.stem for p in new_dir.iterdir() if p.suffix in IMAGE_EXTS}
                 if new_dir.exists() else set())

    # Candidate stems: have a real image + a mappable original fake; prefer those
    # with a new fake (the actual comparison).
    candidates = []
    pool = sorted(new_stems) if (new_stems and not args.allow_missing_new) \
        else sorted(stem2row)
    for stem in pool:
        if stem not in stem2row:
            continue
        if _real_path(real_dir, stem) is None:
            continue
        if _orig_fake_path(orig_dir, prefix, stem2row[stem]) is None:
            continue
        if not args.allow_missing_new and stem not in new_stems:
            continue
        candidates.append(stem)

    if not candidates:
        print(f"[{semantic}] no comparable samples found "
              f"(new_stems={len(new_stems)}, mapped={len(stem2row)}) — skipping")
        return False

    rng = random.Random(f"{args.seed}:{semantic}")
    stems = rng.sample(candidates, min(args.num_samples, len(candidates)))

    cols = ["REAL", f"{gen} (IAB original)", f"{gen} (new caption)"]
    n = len(stems)
    fig, axes = plt.subplots(n, 3, figsize=(3 * 3.2, n * 3.2))
    if n == 1:
        axes = axes.reshape(1, 3)
    for j, title in enumerate(cols):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold")
    for i, stem in enumerate(stems):
        n_row = stem2row[stem]
        paths = [
            _real_path(real_dir, stem),
            _orig_fake_path(orig_dir, prefix, n_row),
            (new_dir / f"{stem}.png") if (new_dir / f"{stem}.png").exists() else None,
        ]
        for j, pth in enumerate(paths):
            ax = axes[i, j]
            ax.axis("off")
            thumb = _load_thumb(pth, args.thumb)
            if thumb is not None:
                ax.imshow(thumb)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=10, color="red")
        axes[i, 0].set_ylabel(stem, fontsize=8)
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

    fig.suptitle(f"{gen} — {semantic}: real vs IAB-original fake vs new-caption fake",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out_dir) / f"compare_{gen}_{semantic}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[{semantic}] saved {out}  ({n} samples)")
    return True


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    any_ok = False
    for sem in args.semantics:
        any_ok |= build_sheet(args, sem)
    if not any_ok:
        sys.exit("No contact sheets produced — check paths/generator.")
    print(f"\nContact sheets in: {args.out_dir}")


if __name__ == "__main__":
    main()
