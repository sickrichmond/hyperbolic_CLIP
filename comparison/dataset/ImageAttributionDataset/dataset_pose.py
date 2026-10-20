import torchvision.transforms as T

from comparison.dataset.ImageAttributionDataset import DATASET
from comparison.dataset.ImageAttributionDataset.dataset import ImageAttributionDataset


@DATASET.register_module(module_name="pose")
class POSEDataset(ImageAttributionDataset):
    def __init__(
        self,
        root_dir,
        num_images_per_semantic_per_class=2000,
        transform=None,
        degraded=0,
        config=None,
        **kwargs,
    ):
        super().__init__(
            root_dir,
            num_images_per_semantic_per_class,
            transform,
            degraded=degraded,
            iabench_source=kwargs.get("iabench_source"),
        )
        if self.transform is None:
            image_size = (config or {}).get("image_size", 128)
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                ]
            )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["image"] = self.transform(item["image"])
        return item
