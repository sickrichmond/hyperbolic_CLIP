"""CLIP preprocessing adapter for the comparison harness.

Inherit ImageAttributionDataset's enumeration, active class map, semantic labels,
grok crop and test degradations. Replace each returned PIL image with the CLIP
processor tensor under the image key. Used by both hyperbolic and spherical
CLIP evaluators.

Optional pre_resize resizes the shortest edge with bicubic interpolation,
preserving aspect ratio, before CLIP processing. It changes the preprocessing
protocol; it does not remove all effects of native image resolution.
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
            # Resize the shortest edge to N while preserving aspect ratio.
            # The processor then resizes/crops; native-resolution effects can remain.
            w, h = image.size
            k = self.pre_resize / min(w, h)
            image = image.resize((max(1, round(w * k)), max(1, round(h * k))),
                                 Image.BICUBIC)
        pixel = self.processor(images=image, return_tensors='pt')['pixel_values'][0]
        item['image'] = pixel
        return item
