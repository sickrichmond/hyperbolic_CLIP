import random

import numpy as np
import torchvision.transforms as T
from PIL import Image

from comparison.dataset.ImageAttributionDataset import DATASET
from .dataset import ImageAttributionDataset


def _low_bit_image(image, size=256, patch_size=32, patch_mode="max", output_mode="full"):
    image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    low_bits = ((image & 0x07) * (255 // 7)).astype(np.uint8)
    result = Image.fromarray(low_bits).resize((size, size), Image.Resampling.BILINEAR)

    if output_mode == "full":
        return result

    patches = [
        np.asarray(result)[y:y + patch_size, x:x + patch_size]
        for y in range(0, size - patch_size + 1, patch_size)
        for x in range(0, size - patch_size + 1, patch_size)
    ]
    if not patches:
        return result

    def complexity(patch):
        return np.abs(np.diff(patch.astype(np.int16), axis=0)).sum() + np.abs(np.diff(patch.astype(np.int16), axis=1)).sum()

    if patch_mode == "min":
        selected = min(patches, key=complexity)
    elif patch_mode == "random":
        selected = random.choice(patches)
    else:
        selected = max(patches, key=complexity)
    return Image.fromarray(selected).resize((size, size), Image.Resampling.BILINEAR)


@DATASET.register_module(module_name="lida")
class LIDADataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None, degraded=0, config=None, **kwargs):
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded,
                         iabench_source=kwargs.get("iabench_source"))
        self.config = config or {}
        self.image_size = self.config.get("image_size", 256)
        self.patch_size = self.config.get("patch_size", 32)
        self.patch_mode = self.config.get("patch_mode", "max")
        self.output_mode = self.config.get("output_mode", "full")
        if self.transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = _low_bit_image(
            item["image"], self.image_size, self.patch_size, self.patch_mode, self.output_mode
        )
        item["image"] = self.transform(image)
        return item
