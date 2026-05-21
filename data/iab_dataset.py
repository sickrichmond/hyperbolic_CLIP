from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPImageProcessor

# Mirrors the SEMANTIC_TO_SUPER mapping from IAB's download.py.
# Flat semantics (COCO, ImageNet-1k) live directly under the generator dir.
# Hierarchical ones are nested under their super-category.
SEMANTIC_TO_SUPER: dict[str, str] = {
    "cat": "AnimalFace",
    "dog": "AnimalFace",
    "wild": "AnimalFace",
    "celebahq": "HumanFace",
    "FFHQ": "HumanFace",
    "bedroom": "Scene",
    "church": "Scene",
    "classroom": "Scene",
    "COCO": "COCO",
    "ImageNet-1k": "ImageNet-1k",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


class IABDataset(Dataset):
    """
    Loads images from a directory populated by IAB's download.py.

    Layout produced by IAB download.py:
        root/
          {generator}/
            {semantic}/              # COCO, ImageNet-1k
              *.jpg
            {SuperCategory}/{semantic}/  # AnimalFace/cat, HumanFace/FFHQ, Scene/bedroom …
              *.jpg

    Args:
        root:           Path to the extracted IAB dataset root.
        generators:     Which generator subdirs to include (e.g. ["FLUX", "real"]).
        semantics:      Which semantic subdirs to include (e.g. ["COCO"]).
        max_per_class:  Cap images per (generator, semantic) pair.  None = no cap.
    """

    def __init__(
        self,
        root: str,
        generators: list[str],
        semantics: list[str],
        processor_name: str = "openai/clip-vit-base-patch32",
        max_per_class: Optional[int] = None,
    ):
        self.root = Path(root)
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)
        self.samples: list[tuple[Path, str]] = []  # (image_path, generator)

        for gen in generators:
            for sem in semantics:
                super_cat = SEMANTIC_TO_SUPER.get(sem, sem)
                if super_cat == sem:
                    img_dir = self.root / gen / sem
                else:
                    img_dir = self.root / gen / super_cat / sem

                if not img_dir.exists():
                    print(f"  [IABDataset] not found, skipping: {img_dir}")
                    continue

                imgs = _images_in(img_dir)
                if max_per_class is not None:
                    imgs = imgs[:max_per_class]
                for p in imgs:
                    self.samples.append((p, gen))

        n_real = sum(1 for _, g in self.samples if g == "real")
        n_fake = len(self.samples) - n_real
        print(
            f"IABDataset: {len(self.samples)} images "
            f"({n_real} real, {n_fake} fake) — "
            f"generators={generators}, semantics={semantics}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, generator = self.samples[idx]
        img = Image.open(path).convert("RGB")
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel,
            "is_real": torch.tensor(generator == "real", dtype=torch.bool),
            "generator": generator,
        }
