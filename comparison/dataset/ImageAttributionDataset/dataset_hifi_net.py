import torchvision.transforms as T
from .dataset import ImageAttributionDataset, hifi_label_mapping
from comparison.dataset.ImageAttributionDataset import DATASET
from torchvision import transforms

@DATASET.register_module(module_name="hifi_net")
class HiFiNetDataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None, degraded=0, **kwargs):
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))
            ])
        # Hierarchical labels [(level1, level2, level3, fine), ...] in the ACTIVE
        # class order — respects IAB_EXCLUDE_GENERATORS (e.g. 22 cls without dalle3).
        # self.model_class_to_label stays the base's (env-aware) map; no override.
        self.label_mapping = hifi_label_mapping()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)  
        image = item["image"]  
        if self.transform:  
            image = self.transform(image)  
        item["image"] = image  
        item["hierachi_label"] = self.label_mapping[item["label"]]
        return item  