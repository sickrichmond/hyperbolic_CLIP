import torchvision.transforms as T

from comparison.dataset.ImageAttributionDataset import DATASET
from .dataset import ImageAttributionDataset


@DATASET.register_module(module_name="defake")
class DeFakeDataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000,
                 transform=None, degraded=0, config=None, **kwargs):
        super().__init__(root_dir, num_images_per_semantic_per_class,
                         transform, degraded=degraded)
        self.image_size = (config or {}).get("image_size", 224)
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["image"] = self.transform(item["image"])
        return item