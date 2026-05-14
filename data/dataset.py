import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler
from PIL import Image
from transformers import CLIPImageProcessor


class OpenFakePairedDataset(Dataset):
    """
    Loads a real/fake subset of OpenFake built by one of:
      - scripts/build_paired_subset.py (full pairing by prompt)
      - scripts/build_from_single_shard.py (single shard, no pairing)

    Expected layout under `root`:
        manifest.parquet      # one row per image
        images/
            *.jpg
    """

    def __init__(
        self,
        root: str,
        processor_name: str = "openai/clip-vit-base-patch32",
        manifest_name: str = "manifest.parquet",
    ):
        self.root = Path(root)
        manifest_path = self.root / manifest_name
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found at {manifest_path}. "
                "Did you run a build_*.py script first?"
            )
        self.manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)

        n_real = (self.manifest["label"] == "real").sum()
        n_fake = (self.manifest["label"] == "fake").sum()
        n_prompts = self.manifest["prompt_id"].nunique()
        print(
            f"OpenFakePairedDataset: {len(self.manifest)} images "
            f"({n_real} real, {n_fake} fake) across {n_prompts} distinct prompt_ids."
        )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img_path = self.root / row["image_path"]
        img = Image.open(img_path).convert("RGB")
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel,
            "is_real": torch.tensor(row["label"] == "real", dtype=torch.bool),
            "prompt_id": torch.tensor(int(row["prompt_id"]), dtype=torch.long),
        }


def make_balanced_sampler(dataset: OpenFakePairedDataset) -> WeightedRandomSampler:
    """
    Balanced random sampler that gives equal probability to real and fake samples.

    Use this when the dataset does not have prompt-level pairing (e.g. a single
    shard of OpenFake). Each batch will be approximately 50/50 real/fake.
    """
    labels = dataset.manifest["label"].values
    n_real = int((labels == "real").sum())
    n_fake = int((labels == "fake").sum())
    if n_real == 0 or n_fake == 0:
        raise RuntimeError(
            f"Dataset is not balanced for sampling: {n_real} real, {n_fake} fake."
        )
    weights = [
        (1.0 / n_real) if l == "real" else (1.0 / n_fake)
        for l in labels
    ]
    return WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True
    )


class PairedPromptBatchSampler(Sampler):
    """
    Batches where every prompt contributes both real and fake samples.
    Use when the manifest has prompt-level pairing (full pre-filtering script).
    """

    def __init__(
        self,
        dataset: OpenFakePairedDataset,
        prompts_per_batch: int = 16,
        reals_per_prompt: int = 1,
        fakes_per_prompt: int = 1,
        num_batches: int | None = None,
        seed: int = 42,
    ):
        super().__init__(data_source=dataset)
        self.dataset = dataset
        self.prompts_per_batch = prompts_per_batch
        self.reals_per_prompt = reals_per_prompt
        self.fakes_per_prompt = fakes_per_prompt
        self.rng = random.Random(seed)

        self.real_by_prompt: dict[int, list[int]] = defaultdict(list)
        self.fake_by_prompt: dict[int, list[int]] = defaultdict(list)
        for i, row in dataset.manifest.iterrows():
            pid = int(row["prompt_id"])
            if row["label"] == "real":
                self.real_by_prompt[pid].append(i)
            else:
                self.fake_by_prompt[pid].append(i)

        self.usable_prompts = [
            pid for pid in self.real_by_prompt
            if pid in self.fake_by_prompt
            and len(self.real_by_prompt[pid]) >= 1
            and len(self.fake_by_prompt[pid]) >= 1
        ]
        if not self.usable_prompts:
            raise RuntimeError(
                "No prompts have both real and fake samples. "
                "Use make_balanced_sampler instead."
            )

        self.batch_size = prompts_per_batch * (reals_per_prompt + fakes_per_prompt)
        if num_batches is None:
            self.num_batches = max(1, len(dataset) // self.batch_size)
        else:
            self.num_batches = num_batches

    def _sample(self, pool: list[int], k: int) -> list[int]:
        if len(pool) < k:
            return self.rng.choices(pool, k=k)
        return self.rng.sample(pool, k=k)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = self.rng.sample(
                self.usable_prompts,
                k=min(self.prompts_per_batch, len(self.usable_prompts)),
            )
            batch: list[int] = []
            for pid in chosen:
                batch.extend(self._sample(self.real_by_prompt[pid], self.reals_per_prompt))
                batch.extend(self._sample(self.fake_by_prompt[pid], self.fakes_per_prompt))
            yield batch

    def __len__(self):
        return self.num_batches