"""
Build a prompted real/fake dataset from OpenFake in one scan, producing two
manifests:

  manifest.parquet        — ALL images that have a non-empty prompt.
                            Used for training phase 1 (L_real + L_push).

  manifest_paired.parquet — Subset of manifest.parquet where the prompt_id
                            appears in BOTH real and fake rows.
                            Used for training phase 2 (L_real + L_push + L_pair).

Both manifests share the same image files in output_dir/images/.
prompt_id is consistent: the same integer for all images sharing the same
prompt text, so PairedPromptBatchSampler can find (real, fake) pairs.

Pass 1 (zero local disk):
    Read only prompt+label from each parquet via HTTP range requests
    (HfFileSystem + pyarrow column projection). Builds a prompt→id map.
    Saves to pass1_stats.parquet so pass1 is skipped if restarted.

Pass 2 (streaming, zero temp disk):
    Stream full dataset, save all images whose prompt is non-empty
    (subject to optional size caps).

Run in tmux — pass1 ~4-5h, pass2 ~6-10h.
"""
from __future__ import annotations

import argparse
import io
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

REPO_ID = "ComplexDataLab/OpenFake"
REAL_LABEL = "real"
FAKE_LABEL = "fake"
MIN_PROMPT_LEN = 10


# ---------------------------------------------------------------------------
# Pass 1 — HTTP column projection
# ---------------------------------------------------------------------------

def list_parquet_files(config: str, split: str) -> list[str]:
    fs = HfFileSystem()
    folder = f"datasets/{REPO_ID}/{config}"
    all_files = fs.ls(folder, detail=False)
    return sorted([
        f for f in all_files
        if f.endswith(".parquet") and f"/{split}-" in f
    ])


def pass1_build_prompt_map(
    parquet_paths: list[str],
    stats_cache: Path,
) -> dict[str, int]:
    """
    Returns prompt → prompt_id mapping for all prompts seen in the dataset.
    Uses HTTP column projection: only prompt+label columns are fetched.
    Results cached to stats_cache to allow restart without re-running pass1.
    """
    if stats_cache.exists():
        print(f"Pass1 cache found at {stats_cache}, loading...")
        df = pd.read_parquet(stats_cache)
        prompt_to_id = {row.prompt: row.prompt_id for row in df.itertuples()}
        print(f"Loaded {len(prompt_to_id)} prompts from cache.")
        return prompt_to_id

    fs = HfFileSystem()
    all_prompts: set[str] = set()
    n_errors = 0

    for path in tqdm(parquet_paths, desc="pass1 (HTTP column projection)"):
        try:
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                table = pf.read(columns=["prompt"])
            for p in table.column("prompt").to_pylist():
                if isinstance(p, str) and len(p.strip()) >= MIN_PROMPT_LEN:
                    all_prompts.add(p.strip())
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"\n  Warning: {path.split('/')[-1]}: {e}")

    if n_errors:
        print(f"Total pass1 errors: {n_errors}")

    prompt_to_id = {p: i for i, p in enumerate(sorted(all_prompts))}
    pd.DataFrame([
        {"prompt": p, "prompt_id": i}
        for p, i in prompt_to_id.items()
    ]).to_parquet(stats_cache, index=False)
    print(f"Saved {len(prompt_to_id)} prompts to {stats_cache}")
    return prompt_to_id


# ---------------------------------------------------------------------------
# Pass 2 — streaming
# ---------------------------------------------------------------------------

def pass2_stream_and_save(
    prompt_to_id: dict[str, int],
    output_dir: Path,
    config: str,
    split: str,
    max_real: int | None,
    max_fake: int | None,
) -> pd.DataFrame:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_real_saved = n_fake_saved = n_errors = 0
    records: list[dict] = []
    img_idx = 0

    print(f"\nStreaming {REPO_ID} ({config}/{split})...")
    ds = load_dataset(REPO_ID, config, split=split, streaming=True)

    for row in tqdm(ds, desc="pass2 (streaming)"):
        # Early stop if both caps are hit
        if (max_real is not None and n_real_saved >= max_real and
                max_fake is not None and n_fake_saved >= max_fake):
            break
        try:
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                continue
            prompt = prompt.strip()
            if prompt not in prompt_to_id:
                continue  # no prompt or too short

            label = row.get("label")
            if label == REAL_LABEL:
                if max_real is not None and n_real_saved >= max_real:
                    continue
            elif label == FAKE_LABEL:
                if max_fake is not None and n_fake_saved >= max_fake:
                    continue
            else:
                continue

            img = row.get("image")
            if not isinstance(img, Image.Image):
                if isinstance(img, dict) and "bytes" in img:
                    img = Image.open(io.BytesIO(img["bytes"]))
                else:
                    continue

            img = img.convert("RGB")
            img.thumbnail((256, 256), Image.LANCZOS)
            fname = f"img_{img_idx:08d}.jpg"
            img.save(images_dir / fname, format="JPEG", quality=80, optimize=True)

            records.append({
                "image_path": f"images/{fname}",
                "prompt": prompt,
                "prompt_id": prompt_to_id[prompt],
                "label": label,
                "model": str(row.get("model") or ""),
                "type": str(row.get("type") or ""),
                "release_date": str(row.get("release_date") or ""),
            })
            img_idx += 1
            if label == REAL_LABEL:
                n_real_saved += 1
            else:
                n_fake_saved += 1

            if img_idx % 5000 == 0:
                _save_manifests(pd.DataFrame(records), output_dir)
                print(f"\n  {img_idx} images saved ({n_real_saved} real, "
                      f"{n_fake_saved} fake)")

        except (OSError, SyntaxError, ValueError) as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"\n  Skipped corrupt image #{n_errors}: {e}")

    if n_errors:
        print(f"Total skipped: {n_errors}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Manifest generation
# ---------------------------------------------------------------------------

def _save_manifests(df: pd.DataFrame, output_dir: Path) -> None:
    """Save manifest.parquet (all) and manifest_paired.parquet (paired only)."""
    df.to_parquet(output_dir / "manifest.parquet", index=False)

    if len(df) == 0:
        return

    # Paired: prompt_ids that appear in both real and fake
    real_pids = set(df.loc[df.label == REAL_LABEL, "prompt_id"])
    fake_pids = set(df.loc[df.label == FAKE_LABEL, "prompt_id"])
    paired_pids = real_pids & fake_pids

    df_paired = df[df.prompt_id.isin(paired_pids)].copy()
    df_paired.to_parquet(output_dir / "manifest_paired.parquet", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="core", choices=["core", "reddit"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_real", type=int, default=None,
                    help="Cap on total real images saved (default: no cap).")
    ap.add_argument("--max_fake", type=int, default=None,
                    help="Cap on total fake images saved (default: no cap).")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_cache = output_dir / "pass1_prompt_map.parquet"

    # --- Pass 1 ---
    print("=== Pass 1: build prompt vocabulary via HTTP column projection ===")
    parquet_files = list_parquet_files(args.config, args.split)
    print(f"Found {len(parquet_files)} parquet files")
    prompt_to_id = pass1_build_prompt_map(parquet_files, stats_cache)
    print(f"Prompts with len >= {MIN_PROMPT_LEN}: {len(prompt_to_id)}")

    # --- Pass 2 ---
    print("\n=== Pass 2: stream and save all prompted images ===")
    manifest = pass2_stream_and_save(
        prompt_to_id=prompt_to_id,
        output_dir=output_dir,
        config=args.config,
        split=args.split,
        max_real=args.max_real,
        max_fake=args.max_fake,
    )

    _save_manifests(manifest, output_dir)

    # --- Summary ---
    n_real = (manifest.label == REAL_LABEL).sum()
    n_fake = (manifest.label == FAKE_LABEL).sum()

    paired_pids = (
        set(manifest.loc[manifest.label == REAL_LABEL, "prompt_id"]) &
        set(manifest.loc[manifest.label == FAKE_LABEL, "prompt_id"])
    )
    df_paired = manifest[manifest.prompt_id.isin(paired_pids)]
    n_paired_real = (df_paired.label == REAL_LABEL).sum()
    n_paired_fake = (df_paired.label == FAKE_LABEL).sum()

    disk_mb = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    ) / 1e6

    print(f"\n=== Done ===")
    print(f"manifest.parquet        : {len(manifest)} images "
          f"({n_real} real, {n_fake} fake)")
    print(f"manifest_paired.parquet : {len(df_paired)} images "
          f"({n_paired_real} real, {n_paired_fake} fake) "
          f"across {len(paired_pids)} prompt_ids")
    print(f"Disk usage: {disk_mb:.0f} MB")


if __name__ == "__main__":
    main()
