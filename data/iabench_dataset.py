"""Image-only loader for the Ant0ny/IABench ModelScope Arrow download.

Pass the snapshot directory (containing ``data/``) or that ``data/``
directory itself. The published dataset contains one train split; this
loader does not create validation or test splits.
"""

from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class IABenchDataset(Dataset):
    def __init__(
        self,
        root: str,
        processor_name: str = "openai/clip-vit-base-patch32",
        generators: list[str] | None = None,
        max_per_class: int | None = None,
    ):
        if max_per_class is not None and max_per_class < 0:
            raise ValueError("max_per_class must be non-negative")

        root_path = Path(root)
        data_dir = (root_path / "data" if (root_path / "data" / "state.json").is_file()
                    else root_path)
        if not (data_dir / "state.json").is_file():
            raise FileNotFoundError(
                f"IABench Arrow dataset not found in {root_path} (expected data/state.json)")

        self.data = load_from_disk(str(data_dir), keep_in_memory=False)
        missing = {"image", "label", "file_name"} - set(self.data.column_names)
        if missing:
            raise ValueError(f"IABench dataset is missing columns: {sorted(missing)}")

        if generators is None and max_per_class is None:
            self.indices = range(len(self.data))
        else:
            requested = set(generators) if generators is not None else None
            counts: Counter[str] = Counter()
            seen: set[str] = set()
            self.indices = []
            for index, label in enumerate(self.data["label"]):
                seen.add(label)
                if requested is not None and label not in requested:
                    continue
                if max_per_class is not None and counts[label] >= max_per_class:
                    continue
                self.indices.append(index)
                counts[label] += 1
            if requested is not None and requested - seen:
                raise ValueError(f"IABench has no labels: {sorted(requested - seen)}")

        self.processor = CLIPImageProcessor.from_pretrained(processor_name)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row = self.data[self.indices[index]]
        label = row["label"]
        pixel = self.processor(images=row["image"].convert("RGB"),
                               return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel,
            "generator": label,
            "is_real": torch.tensor(label.casefold() == "real", dtype=torch.bool),
            "file_name": row["file_name"],
        }
