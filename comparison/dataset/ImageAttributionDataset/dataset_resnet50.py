import torchvision.transforms as T  
from .dataset import ImageAttributionDataset
from comparison.dataset.ImageAttributionDataset import DATASET
from data.degradations import random_degradation


@DATASET.register_module(module_name='resnet50')
class Resnet50Dataset(ImageAttributionDataset):  
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000, transform=None,degraded=0, **kwargs):  
        super().__init__(root_dir, num_images_per_semantic_per_class, transform, degraded=degraded, iabench_source=kwargs.get("iabench_source"))
        self.train_augment = (self.iabench_source is not None and
                              (kwargs.get('config') or {}).get('train_augment', False))
        if self.transform is None:  
            self.transform = T.Compose([  
                T.Resize(256),  
                T.CenterCrop(224),  
                T.ToTensor(),  
                T.Normalize(mean=[0.485, 0.456, 0.406],  
                            std=[0.229, 0.224, 0.225]),  
            ])  

    def __getitem__(self, idx):  
        item = super().__getitem__(idx)  
        image = item['image']  
        if self.train_augment and self.mode == 'train':
            image = random_degradation(image)
        if self.transform:  
            image = self.transform(image)  
        item['image'] = image  
        return item  