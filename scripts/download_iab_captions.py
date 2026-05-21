"""
Download IAB caption CSV files from the public GitHub repository.

The captions are Qwen-VL-Chat descriptions of the real images.
Synthetic images inherit the same caption (they were generated from it).

Usage:
    python -m scripts.download_iab_captions \
        --output_dir /mnt/data3/rtrebiani/iab_captions

Downloads 10 CSV files (~few MB total), one per semantic class.
"""
import argparse
from pathlib import Path

import requests

REPO_BASE = (
    "https://raw.githubusercontent.com/mttry/ImageAttributionBench"
    "/clean-main/dataset_construction/prompt_generator"
    "/downloaded_captions/final_captions_new"
)

# Real-image CSVs (stem-based lookup: filename stem → caption)
REAL_CAPTION_FILES = {
    "COCO.csv":                 "COCO",
    "AnimalFace_cat.csv":       "cat",
    "AnimalFace_dog.csv":       "dog",
    "AnimalFace_wild.csv":      "wild",
    "HumanFace_FFHQ.csv":       "FFHQ",
    "HumanFace_celebahq.csv":   "celebahq",
    "Scene_LSUN_bedroom.csv":   "bedroom",
    "Scene_LSUN_church.csv":    "church",
    "Scene_LSUN_classroom.csv": "classroom",
    "imagenet-1k.csv":          "ImageNet-1k",
}

# Synthetic-image CSVs (index-based lookup: _p{N}_i{K} → row N)
# COCO-new and imagenet-1k-new are the prompt sets used to generate fakes.
FAKE_CAPTION_FILES = {
    "COCO-new.csv",
    "imagenet-1k-new.csv",
}

CAPTION_FILES = {**REAL_CAPTION_FILES, **{f: f for f in FAKE_CAPTION_FILES}}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  already exists, skipping: {dest.name}")
        return
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_bytes(r.content)
    kb = len(r.content) / 1024
    print(f"  {dest.name:35s}  {kb:7.1f} KB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_dir", default="/mnt/data3/rtrebiani/iab_captions")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(CAPTION_FILES)} caption CSVs → {out}\n")
    for fname in CAPTION_FILES:
        url = f"{REPO_BASE}/{fname}"
        download(url, out / fname)

    print(f"\nDone. Files saved to {out}")


if __name__ == "__main__":
    main()
