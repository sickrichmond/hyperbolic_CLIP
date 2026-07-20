import torchvision.transforms as T  
from .dataset import ImageAttributionDataset, hifi_label_mapping
from comparison.dataset.ImageAttributionDataset import DATASET
from torchvision import transforms
import clip

@DATASET.register_module(module_name="defl")  
class DEFLDataset(ImageAttributionDataset):  
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
        _, self.clip_preprocess = clip.load("RN50x16", device="cuda")
        # level0: 0 generated, 1 real;
        # level1: 0 commercial, 1 open-source, 2 real;
        # level2: 0 commercial, 1 SD, 2 diffusers, 3 DiT, 4 AR, 5 real;
        # level3: the same as label
        # Derived from the ACTIVE label map (respects IAB_EXCLUDE_GENERATORS) — the
        # taxonomy is identical to HiFi-Net's, so the same helper serves both. The
        # 23 hardcoded tuples this replaces mis-grouped every class from index 11 on
        # once dalle3 was excluded (hidream got 'commercial' instead of 'DiT'),
        # corrupting method_label and hence the dual-margin contrastive loss.
        self.label_mapping = hifi_label_mapping()
    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item["image"]  
        clip_image = self.clip_preprocess(image)
        if self.transform:  
            image = self.transform(image)  
        item["image"] = image  
        item["clip_image"] = clip_image  
        item["method_label"] = self.label_mapping[item["label"]][2]
        return item  