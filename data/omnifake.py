"""The ten OmniFake test classes that correspond to IAB checkpoint labels."""

from pathlib import Path


OMNIFAKE_TO_IAB = {
    'GPT4_o': '4o',
    'Hunyuan_DiT': 'hunyuan',
    'Infinity': 'infinity',
    'Janus_Pro': 'janus-pro',
    'Midjourney_V_6': 'mid-6.0',
    'PixArt_Alpha': 'PIXART',
    'Playground_V_2_5': 'PLAYGROUND_2_5',
    'SDXL': 'SDXL',
    'Stable_Diffusion_3_Medium': 'SD3',
    'Stable_Diffusion_V_1_5': 'SD1_5',
}
IMAGES_PER_GENERATOR = 2000
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png'}


def discover_test_samples(root):
    samples = []
    for folder, label in OMNIFAKE_TO_IAB.items():
        directory = Path(root) / 'val' / folder
        images = sorted(path for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(images) != IMAGES_PER_GENERATOR:
            raise ValueError(f'{directory}: expected {IMAGES_PER_GENERATOR} test images, '
                             f'found {len(images)}')
        samples.extend((path, label) for path in images)
    return samples
