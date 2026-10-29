"""Corrected IABench cone-training taxonomy, independent of baseline groupings.

Based on https://github.com/Ant0ny44/IncreFA/blob/main/configs/iabench_families.json
CogView4 is diffusion (https://github.com/zai-org/CogView4). Midjourney and
Nano Banana share an operational unknown group, not a claim of shared ancestry.
The legacy increfa option/function names are retained for CLI compatibility.
"""
import math


INCREFA_FAMILIES = {
    "real": ("real",),
    "gan": ("E4S", "R3GAN", "StyleGAN-XL", "stylegan3"),
    "autoregressive": ("gpt4o",),
    "diffusion": (
        "cogview4",
        "CompVis@ldm-text2im-large-256", "CompVis@stable-diffusion-v1-4",
        "Imagen3", "PixArt-alpha@PixArt-XL-2-1024-MS",
        "black-forest-labs@FLUX.1-dev", "black-forest-labs@FLUX.1-schnell",
        "latent-consistency@lcm-lora-sdv1-5", "latent-consistency@lcm-lora-sdxl",
        "playgroundai@playground-v2-1024px-aesthetic",
        "playgroundai@playground-v2.5-1024px-aesthetic",
        "runwayml@stable-diffusion-v1-5", "segmind@SSD-1B",
        "segmind@SegMoE-SD-4x2-v0", "segmind@small-sd", "segmind@tiny-sd",
        "stabilityai@sdxl-turbo", "stabilityai@stable-diffusion-2",
        "stabilityai@stable-diffusion-2-1",
        "stabilityai@stable-diffusion-3-medium-diffusers",
        "stabilityai@stable-diffusion-xl-base-1.0",
    ),
    "unknown": ("midjourneyv6", "nano-banana"),
}


def increfa_mapping(class_names, family_groups=None):
    """Return active families and indices aligned with the exact class order."""
    groups = INCREFA_FAMILIES if family_groups is None else family_groups
    parent = {name: family for family, names in groups.items() for name in names}
    if len(parent) != sum(len(names) for names in groups.values()):
        raise ValueError("A generator cannot belong to multiple families")
    missing = [name for name in class_names if name not in parent]
    if missing:
        raise ValueError(f"IncreFA taxonomy has no family for {missing}")
    active = {parent[name] for name in class_names}
    families = [family for family in groups if family in active]
    return families, [families.index(parent[name]) for name in class_names]


def minimum_cone_depth(curv, min_radius):
    """Tangent radius safely outside the aperture formula's excluded ball."""
    return math.asinh(2 * min_radius) / math.sqrt(curv) + 1e-3
