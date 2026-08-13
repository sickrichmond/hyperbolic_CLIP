"""
Harness-native dataset adapter for the hyperbolic-CLIP attributor.

Subclasses the baseline `ImageAttributionDataset` so it inherits the EXACT same
image enumeration, 23-class label map, semantic labels, grok crop and test-time
degradation. Because `get_dataloader` performs its stratified split on
`dataset.samples` (identical order across all baseline datasets), evaluating the
hyperbolic model through this adapter uses BYTE-IDENTICAL test images / split /
degradations as resnet50/dct/hifi_net/defl.

The only difference from the baselines is that `__getitem__` returns a
CLIP-preprocessed `pixel_values` tensor (under key 'image') instead of a PIL
image, so our model can consume it directly.

`config['pre_resize']` is the one deliberate deviation, and it is a CONTROL, not a
default (see test_hypclip.py --pre_resize): it squares every image to a common size
before the CLIP processor, so the native→224 resampling ratio stops varying by
class. tests/audit_shortcuts.py shows that ratio is strongly class-dependent, and
this is how we find out whether the model was reading it.
"""
from PIL import Image

from .dataset import ImageAttributionDataset
from comparison.dataset.ImageAttributionDataset import DATASET
from transformers import CLIPImageProcessor


@DATASET.register_module(module_name='hypclip')
class HypclipDataset(ImageAttributionDataset):
    def __init__(self, root_dir, num_images_per_semantic_per_class=2000,
                 transform=None, degraded=0, config=None, **kwargs):
        super().__init__(root_dir, num_images_per_semantic_per_class,
                         transform, degraded=degraded)
        cfg = config or {}
        processor_name = cfg.get('clip_name',
                                 cfg.get('processor_name',
                                         'openai/clip-vit-large-patch14'))
        self.processor = CLIPImageProcessor.from_pretrained(processor_name)
        self.pre_resize = cfg.get('pre_resize') or 0

    def __getitem__(self, idx):
        # super() applies the grok crop and (test mode) the degradation, returning
        # a PIL RGB image — same pipeline as every baseline.
        item = super().__getitem__(idx)
        image = item['image']
        if self.pre_resize:
            # Squared, so aspect ratio is equalised too: both channels the audit
            # flags as surviving the preprocessing die at once.
            image = image.resize((self.pre_resize, self.pre_resize), Image.BICUBIC)
        pixel = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
        item['image'] = pixel
        return item
