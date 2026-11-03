"""Shared paper augmentation, followed by OmniDFA local/global image views."""
from PIL import Image
from torchvision import transforms as T

from comparison.dataset.ImageAttributionDataset import DATASET
from comparison.dataset.ImageAttributionDataset.dataset import ImageAttributionDataset
from data.degradations import omnidfa_augment


@DATASET.register_module(module_name='omnidfa')
class OmniDFADataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000,
                 transform=None, degraded=0, **kwargs):
        if transform is not None:
            raise ValueError('OmniDFA owns its two-view preprocessing')
        super().__init__(root_dir, num_images_per_semantic_per_class,
                         degraded=degraded, iabench_source=kwargs.get('iabench_source'))
        self.augment = omnidfa_augment
        self.local_train = T.RandomCrop(224)
        self.global_train = T.Compose([T.Resize(224), T.RandomCrop(224)])
        self.local_eval = T.CenterCrop(224)
        self.global_eval = T.Compose([T.Resize(224), T.CenterCrop(224)])
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = item['image']
        if self.mode == 'train':
            image = self.augment(image)
            # The shared scale augmentation can make images too small for a 224 crop.
            scale = max(1.0, 224 / min(image.size))
            if scale > 1:
                image = image.resize(tuple(round(side * scale) for side in image.size),
                                     Image.Resampling.BICUBIC)
            local, global_image = self.local_train(image), self.global_train(image)
        else:
            # Upstream also randomizes evaluation crops/interpolation; fix both here.
            scale = max(1.0, 256 / min(image.size))
            if scale > 1:
                image = image.resize(tuple(round(side * scale) for side in image.size),
                                     Image.Resampling.BILINEAR)
            local, global_image = self.local_eval(image), self.global_eval(image)
        item['image'] = self.to_tensor(local)
        item['global_image'] = self.to_tensor(global_image)
        return item
