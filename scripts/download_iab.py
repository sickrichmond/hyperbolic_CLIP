"""
Download a subset of ImageAttributionBench from Harvard Dataverse.

The full dataset (>500 GB) is hosted at:
    https://dataverse.harvard.edu  doi:10.7910/DVN/O4S4IV

This script downloads only the generators/semantics you request, extracts
images to the canonical IAB layout, and optionally deletes the zip files.

Canonical output layout:
    <dataset_path>/
      FLUX/
        COCO/
          *.jpg
      real/
        COCO/
          *.jpg
      ...

Usage (minimal test — FLUX + real, COCO only):
    python -m scripts.download_iab \\
        --dataset_path /mnt/data3/rtrebiani/iab_dataset \\
        --model_classes FLUX real \\
        --semantic_classes COCO \\
        --delete_zip

Harvard Dataverse API token (optional, speeds up downloads):
    export DATAVERSE_TOKEN=<your_token>
    Register free at https://dataverse.harvard.edu

Requirements:  requests, tqdm, p7zip (apt install p7zip-full)
"""
import argparse
import os
import shutil
import subprocess
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PERSISTENT_ID = "doi:10.7910/DVN/O4S4IV"
BASE_API = "https://dataverse.harvard.edu"

ALL_MODEL_CLASSES = [
    "4o", "CogView3_PLUS", "FLUX", "KANDINSKY", "PIXART", "PLAYGROUND_2_5",
    "SD1_5", "SD2_1", "SD3", "SD3_5", "SDXL", "dalle3", "gemini", "grok3",
    "hidream", "hunyuan", "ideogram", "infinity", "janus-pro", "kling",
    "mid-5.2", "mid-6.0", "real",
]
ALL_SEMANTIC_CLASSES = [
    "COCO", "FFHQ", "ImageNet-1k", "bedroom", "cat", "celebahq",
    "church", "classroom", "dog", "wild",
]
SEMANTIC_TO_SUPER = {
    "cat": "AnimalFace", "dog": "AnimalFace", "wild": "AnimalFace",
    "celebahq": "HumanFace", "FFHQ": "HumanFace",
    "bedroom": "Scene", "church": "Scene", "classroom": "Scene",
    "COCO": "COCO", "ImageNet-1k": "ImageNet-1k",
}
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", required=True,
                   help="Directory where extracted images will be stored.")
    p.add_argument("--download_path", default=None,
                   help="Directory for temporary zip files. Defaults to <dataset_path>/.zips")
    p.add_argument("--model_classes", nargs="+", choices=ALL_MODEL_CLASSES,
                   default=["FLUX", "real"])
    p.add_argument("--semantic_classes", nargs="+", choices=ALL_SEMANTIC_CLASSES,
                   default=["COCO"])
    p.add_argument("--delete_zip", action="store_true",
                   help="Delete zip files after extraction.")
    p.add_argument("--no_parallel", action="store_true",
                   help="Disable parallel chunk download.")
    p.add_argument("--num_threads", type=int, default=8,
                   help="Number of download threads per file.")
    return p.parse_args()


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_sequential(url: str, save_path: Path, token: str) -> None:
    headers = {"X-Dataverse-key": token} if token else {}
    with requests.get(url, headers=headers, stream=True, verify=False) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
            desc=save_path.name, total=total, unit="iB", unit_scale=True
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))


def _download_parallel(url: str, save_path: Path, token: str, num_threads: int) -> None:
    headers = {"X-Dataverse-key": token} if token else {}
    with requests.get(url, headers=headers, stream=True, verify=False) as r:
        r.raise_for_status()
        actual_url = r.url
        total = int(r.headers.get("content-length", 0))
        accepts_ranges = r.headers.get("accept-ranges", "").lower() == "bytes"

    if not total or not accepts_ranges:
        _download_sequential(actual_url, save_path, token)
        return

    chunk = total // num_threads
    part_paths = [Path(f"{save_path}.part{i}") for i in range(num_threads)]
    lock = threading.Lock()

    def _fetch(start: int, end: int, part: Path, bar: tqdm) -> None:
        with requests.get(actual_url, headers={"Range": f"bytes={start}-{end}"},
                          stream=True, verify=False) as resp:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for data in resp.iter_content(chunk_size=65536):
                    f.write(data)
                    with lock:
                        bar.update(len(data))

    with tqdm(desc=save_path.name, total=total, unit="iB", unit_scale=True) as bar:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futs = [
                ex.submit(_fetch,
                          i * chunk,
                          total - 1 if i == num_threads - 1 else (i + 1) * chunk - 1,
                          part_paths[i], bar)
                for i in range(num_threads)
            ]
            for f in as_completed(futs):
                f.result()

    with open(save_path, "wb") as out:
        for pp in part_paths:
            with open(pp, "rb") as inp:
                shutil.copyfileobj(inp, out)
            pp.unlink()


# ── Core logic ────────────────────────────────────────────────────────────────

def fetch_file_list(token: str) -> list[dict]:
    url = (f"{BASE_API}/api/datasets/:persistentId/versions/:latest/files"
           f"?persistentId={PERSISTENT_ID}")
    headers = {"X-Dataverse-key": token} if token else {}
    print(f"Fetching file list from Dataverse…")
    r = requests.get(url, headers=headers, verify=False, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Dataverse API error {r.status_code}:\n{r.text}")
    files = r.json().get("data", [])
    print(f"  {len(files)} files found in dataset.")
    return files


def build_groups(files: list[dict], model_classes: list[str],
                 semantic_classes: list[str]) -> dict[str, list[dict]]:
    valid = {}
    for m in ALL_MODEL_CLASSES:
        for s in ALL_SEMANTIC_CLASSES:
            key = f"{m}_{s}".lower()
            valid[key] = (m, s)
            if "-" in s:
                valid[f"{m}_{s.replace('-', '_')}".lower()] = (m, s)

    groups: dict[str, list[dict]] = defaultdict(list)
    for fi in files:
        name = fi["dataFile"]["filename"]
        base, ext = name.rsplit(".", 1)
        if not (ext.lower() == "zip" or ext.lower().startswith("z")):
            continue
        match = valid.get(base.lower())
        if not match:
            continue
        m, s = match
        if m in model_classes and s in semantic_classes:
            groups[base].append(fi)
    return dict(groups)


def extract_and_route(zip_path: Path, model_name: str, semantic_name: str,
                      dataset_path: Path) -> int:
    super_cat = SEMANTIC_TO_SUPER.get(semantic_name, semantic_name)
    if super_cat == semantic_name:
        dest = dataset_path / model_name / semantic_name
    else:
        dest = dataset_path / model_name / super_cat / semantic_name
    dest.mkdir(parents=True, exist_ok=True)

    sandbox = dataset_path / f"_tmp_{model_name}_{semantic_name}"
    sandbox.mkdir(parents=True, exist_ok=True)

    # Use Python's zipfile (no external tools needed).
    # Fall back to 7z only for split archives (.z01, .z02 … siblings present).
    siblings = list(zip_path.parent.glob(zip_path.stem + ".z[0-9]*"))
    use_7z = bool(siblings) or not zipfile.is_zipfile(zip_path)
    if use_7z:
        subprocess.run(["7z", "x", "-y", f"-o{sandbox}", str(zip_path)], check=True)
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(sandbox)

    count = 0
    for root, _, fnames in os.walk(sandbox):
        for fname in fnames:
            if fname.lower().endswith(IMAGE_EXTS):
                src = Path(root) / fname
                src.replace(dest / fname)
                count += 1

    shutil.rmtree(sandbox)
    return count


def main():
    args = parse_args()
    token = os.environ.get("DATAVERSE_TOKEN", "")
    dataset_path = Path(args.dataset_path)
    download_path = Path(args.download_path) if args.download_path else dataset_path / ".zips"
    dataset_path.mkdir(parents=True, exist_ok=True)
    download_path.mkdir(parents=True, exist_ok=True)

    files = fetch_file_list(token)
    groups = build_groups(files, args.model_classes, args.semantic_classes)

    if not groups:
        print("No matching files found. Check --model_classes and --semantic_classes.")
        return

    print(f"\nWill download {len(groups)} archive(s):")
    for name in sorted(groups):
        print(f"  {name}")

    valid_combinations = {}
    for m in ALL_MODEL_CLASSES:
        for s in ALL_SEMANTIC_CLASSES:
            valid_combinations[f"{m}_{s}".lower()] = (m, s)

    for base_name, parts in sorted(groups.items()):
        parts.sort(key=lambda x: x["dataFile"]["filename"])
        model_name, semantic_name = valid_combinations[base_name.lower()]
        print(f"\n[{base_name}] Downloading {len(parts)} part(s)…")

        downloaded = []
        main_zip = None
        for fi in parts:
            fname = fi["dataFile"]["filename"]
            save_path = download_path / fname
            dl_url = f"{BASE_API}/api/access/datafile/{fi['dataFile']['id']}"

            if save_path.exists():
                print(f"  Already exists, skipping: {fname}")
            else:
                if args.no_parallel:
                    _download_sequential(dl_url, save_path, token)
                else:
                    _download_parallel(dl_url, save_path, token, args.num_threads)

            downloaded.append(save_path)
            if fname.lower().endswith(".zip"):
                main_zip = save_path

        if main_zip is None:
            print(f"  ERROR: no .zip found for {base_name}, skipping.")
            continue

        print(f"  Extracting → {dataset_path / model_name}…")
        n = extract_and_route(main_zip, model_name, semantic_name, dataset_path)
        print(f"  {n} images extracted.")

        if args.delete_zip:
            for p in downloaded:
                p.unlink(missing_ok=True)
            print(f"  Zip(s) deleted.")

    print("\nDone.")


if __name__ == "__main__":
    main()
