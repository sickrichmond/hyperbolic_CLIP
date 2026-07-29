"""Harness-native adapter for the multi-view attributor, native patch source.

Same as `dataset_hypclip` (identical enumeration, split, grok crop and test-time
degradation, so the test images stay byte-identical to every baseline), except
`__getitem__` returns the whole image PLUS a 3x3 grid of half-size windows cut at
FULL resolution, all CLIP-preprocessed: 'image' is (10, 3, 224, 224).

Registered separately instead of adding a flag to `hypclip`: that adapter feeds
`test_hypclip.py` and the numbers already in the comparison tables, and must not
change behaviour.
"""
from .dataset import ImageAttributionDataset
from comparison.dataset.ImageAttributionDataset import DATASET
from transformers import CLIPImageProcessor

from data.iab_clip_dataset import native_patch_grid


@DATASET.register_module(module_name='hypclip_patch')
class HypclipPatchDataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000,
                 transform=None, degraded=0, config=None, **kwargs):
        super().__init__(root_dir, num_images_per_semantic_per_class,
                         transform, degraded=degraded)
        cfg = config or {}
        processor_name = cfg.get('clip_name',
                                 cfg.get('processor_name',
                                         'openai/clip-vit-large-patch14'))
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # super() has applied the grok crop and, in test mode, the degradation —
        # so the windows are cut from the SAME degraded image the baselines see.
        item['image'] = self.processor(images=native_patch_grid(item['image']),
                                       return_tensors='pt')['pixel_values']
        return item
