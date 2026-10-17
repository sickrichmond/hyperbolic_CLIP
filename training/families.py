"""IncreFA's released IABench taxonomy, independent of baseline groupings.

Source: https://github.com/Ant0ny44/IncreFA/blob/main/configs/iabench_families.json
These are the upstream experimental assignments, not verified model ancestry.
"""
import math


INCREFA_FAMILIES = {
    "real": ("real",),
    "gan": ("E4S", "R3GAN", "StyleGAN-XL", "stylegan3"),
    "autoregressive": ("cogview4", "gpt4o"),
    "diffusion": (
        "CompVis@ldm-text2im-large-256", "CompVis@stable-diffusion-v1-4",
        "Imagen3", "PixArt-alpha@PixArt-XL-2-1024-MS",
        "black-forest-labs@FLUX.1-dev", "black-forest-labs@FLUX.1-schnell",
        "latent-consistency@lcm-lora-sdv1-5", "latent-consistency@lcm-lora-sdxl",
        "midjourneyv6", "nano-banana",
        "playgroundai@playground-v2-1024px-aesthetic",
        "playgroundai@playground-v2.5-1024px-aesthetic",
        "runwayml@stable-diffusion-v1-5", "segmind@SSD-1B",
        "segmind@SegMoE-SD-4x2-v0", "segmind@small-sd", "segmind@tiny-sd",
        "stabilityai@sdxl-turbo", "stabilityai@stable-diffusion-2",
        "stabilityai@stable-diffusion-2-1",
        "stabilityai@stable-diffusion-3-medium-diffusers",
        "stabilityai@stable-diffusion-xl-base-1.0",
    ),
}


def increfa_mapping(class_names):
    """Return active families and indices aligned with the exact class order."""
    parent = {name: family for family, names in INCREFA_FAMILIES.items() for name in names}
    missing = [name for name in class_names if name not in parent]
    if missing:
        raise ValueError(f"IncreFA taxonomy has no family for {missing}")
    active = {parent[name] for name in class_names}
    families = [family for family in INCREFA_FAMILIES if family in active]
    return families, [families.index(parent[name]) for name in class_names]


def minimum_cone_depth(curv, min_radius):
    """Tangent radius safely outside the aperture formula's excluded ball."""
    return math.asinh(2 * min_radius) / math.sqrt(curv) + 1e-3
