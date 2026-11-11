"""IAB and IABench cone-training taxonomies, independent of baseline groupings.

The IABench map adapts https://github.com/Ant0ny44/IncreFA/blob/main/configs/iabench_families.json
and uses the diffusion label for CogView4 (https://github.com/zai-org/CogView4).
The unknown groups are shared training labels for models of unclear architecture.
"""
import math


IAB_FAMILIES = {
    "real": ("real",),
    "autoregressive": ("4o", "grok3", "infinity", "janus-pro"),
    "diffusion": (
        "CogView3_PLUS", "FLUX", "KANDINSKY", "PIXART", "PLAYGROUND_2_5",
        "SD1_5", "SD2_1", "SD3", "SD3_5", "SDXL", "dalle3", "hidream", "hunyuan",
    ),
    "unknown": ("gemini", "ideogram", "kling", "mid-5.2", "mid-6.0"),
}


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


def increfa_mapping(class_names, family_groups=None, *, dataset="iabench"):
    """Return active families and indices aligned with the exact class order."""
    groups = family_groups
    if groups is None:
        groups = IAB_FAMILIES if dataset == "iab" else INCREFA_FAMILIES
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
