"""Image-only loader and reproducible partitions for Ant0ny/IABench.

Accept the ModelScope snapshot root or its ``data/`` directory. Images remain
memory mapped. Split manifests identify rows by Arrow index and bind them to
the ordered label/filename metadata; no images are decoded to create a split.
Both saved datasets with JSON metadata and directories of Arrow shards are supported.
"""
from collections import Counter
from copy import copy
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from datasets import Dataset as ArrowDataset, concatenate_datasets, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

from data.degradations import AUG_POLICIES, apply_degradation
from data.image_io import retry_image_read


def manifest_digest(manifest: dict) -> str:
    """Digest both split settings and exact membership, independent of file path."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class IABenchDataset(Dataset):
    def __init__(self, root: str,
                 processor_name: str = "openai/clip-vit-base-patch32",
                 generators: list[str] | None = None,
                 max_per_class: int | None = None):
        if max_per_class is not None and max_per_class < 0:
            raise ValueError("max_per_class must be non-negative")
        root_path = Path(root)
        data_dir = root_path / "data" if (root_path / "data").is_dir() else root_path
        if (data_dir / "state.json").is_file():
            self.data = load_from_disk(str(data_dir), keep_in_memory=False)
        else:
            shards = sorted(data_dir.glob("data-*-of-*.arrow"))
            if not shards:
                raise FileNotFoundError(f"No IABench Arrow shards found in {data_dir}")
            total = int(shards[0].stem.rsplit("-", 1)[-1])
            if len(shards) != total or any(
                    p.name != f"data-{i:05d}-of-{total:05d}.arrow" for i, p in enumerate(shards)):
                raise FileNotFoundError(
                    f"Incomplete IABench Arrow shards in {data_dir}: "
                    f"expected {total} numbered shards, found {len(shards)}")
            self.data = concatenate_datasets([
                ArrowDataset.from_file(str(path), in_memory=False) for path in shards])
        missing = {"image", "label", "file_name"} - set(self.data.column_names)
        if missing:
            raise ValueError(f"IABench dataset is missing columns: {sorted(missing)}")
        self.source_labels = list(self.data["label"])
        self.file_names = list(self.data["file_name"])
        digest = hashlib.sha256()
        for label, name in zip(self.source_labels, self.file_names):
            if not isinstance(label, str) or not label or not isinstance(name, str):
                raise ValueError("IABench labels and filenames must be strings with nonempty labels")
            digest.update((json.dumps([label, name], ensure_ascii=False) + "\n").encode())
        self.dataset_digest = digest.hexdigest()
        self.class_names = sorted(set(self.source_labels))
        requested = set(generators) if generators is not None else set(self.class_names)
        if requested - set(self.class_names):
            raise ValueError(f"IABench has no labels: {sorted(requested - set(self.class_names))}")
        self.class_names = sorted(requested)
        self.max_per_class = max_per_class
        counts = Counter()
        indices = []
        for index, label in enumerate(self.source_labels):
            if label in requested and (max_per_class is None or counts[label] < max_per_class):
                indices.append(index)
                counts[label] += 1
        self._set_indices(indices)
        self.split = "all"
        self.train_augment = False
        self.aug_policy = "corruption"
        self.degraded = 0
        self.pre_resize = 0
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)

    def _set_indices(self, indices):
        self.indices = indices
        self.samples = [(i, self.source_labels[i], self.file_names[i]) for i in indices]

    def make_split_manifest(self, generators=None, max_per_class=None, seed=42,
                            val_frac=0.1, test_frac=0.1) -> dict:
        max_per_class = self.max_per_class if max_per_class is None else max_per_class
        if (not all(math.isfinite(x) and 0 < x < 1 for x in (val_frac, test_frac))
                or val_frac + test_frac >= 1):
            raise ValueError("Split fractions must be positive and val_frac + test_frac < 1")
        if max_per_class is not None and max_per_class <= 0:
            raise ValueError("max_per_class must be positive for splitting")
        names = list(generators) if generators is not None else list(self.class_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("Select nonempty, unique generator names")
        missing = set(names) - set(self.class_names)
        if missing:
            raise ValueError(f"IABench has no labels: {sorted(missing)}")
        real = [n for n in names if n.casefold() == "real"]
        if len(real) > 1:
            raise ValueError(f"Ambiguous real labels: {real}")
        names = real + [n for n in names if n.casefold() != "real"]
        pooled = {n: [] for n in names}
        for i, label in enumerate(self.source_labels):
            if label in pooled and (max_per_class is None or len(pooled[label]) < max_per_class):
                pooled[label].append(i)
        manifest = {"format": "iabench-splits-v1", "dataset_digest": self.dataset_digest,
                    "class_names": names, "seed": seed, "val_frac": val_frac,
                    "test_frac": test_frac, "max_per_class": max_per_class,
                    "train": [], "val": [], "test": []}
        for name, indices in pooled.items():
            random.Random(f"{seed}:{name}").shuffle(indices)
            n_train = math.floor(len(indices) * (1 - (val_frac + test_frac)))
            n_val = math.floor(len(indices) * val_frac)
            if min(n_train, n_val, len(indices) - n_train - n_val) < 1:
                raise ValueError(f"Class {name!r} is too small for three nonempty splits")
            manifest["train"].extend(indices[:n_train])
            manifest["val"].extend(indices[n_train:n_train + n_val])
            manifest["test"].extend(indices[n_train + n_val:])
        return manifest

    def split_view(self, manifest: dict, split: str):
        """Return a view sharing the Arrow data and processor, after validation."""
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        expected = self.make_split_manifest(
            generators=manifest["class_names"], seed=manifest["seed"],
            val_frac=manifest["val_frac"], test_frac=manifest["test_frac"],
            max_per_class=manifest["max_per_class"])
        if manifest != expected:
            raise ValueError("IABench split manifest does not match dataset metadata or membership")
        view = copy(self)
        view._set_indices(list(manifest[split]))
        view.class_names = list(manifest["class_names"])
        view.max_per_class = manifest["max_per_class"]
        view.split = split
        view.train_augment = self.train_augment and split == "train"
        return view

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row_index, label, name = self.samples[index]
        image = retry_image_read(lambda: self.data[row_index]["image"].convert("RGB"))
        if self.degraded and self.train_augment:
            raise ValueError("train_augment and degraded are mutually exclusive")
        if self.degraded:
            image = apply_degradation(image, self.degraded)
        elif self.train_augment:
            image = AUG_POLICIES[self.aug_policy](image)
        if self.pre_resize:
            scale = self.pre_resize / min(image.size)
            image = image.resize(tuple(max(1, round(d * scale)) for d in image.size), Image.BICUBIC)
        pixel = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel, "generator": label,
                "is_real": torch.tensor(label.casefold() == "real", dtype=torch.bool),
                "file_name": name}
