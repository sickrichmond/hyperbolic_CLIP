"""
Image degradations for robustness evaluation.

These replicate EXACTLY the pipeline used by the ImageAttributionBench comparison
methods (comparison/dataset/ImageAttributionDataset/dataset.py -> get_degraded_img),
so our hyperbolic model is tested under identical corruptions and the robustness
curves are directly comparable:

  level 0: clean (no change)
  level 1: downsample x0.5   (NEAREST down- then up-scale)
  level 2: downsample x0.25
  level 3: JPEG compression, quality 65
  level 4: JPEG compression, quality 30
  level 5: Gaussian blur, sigma 3
  level 6: Gaussian blur, sigma 5

NOTE: downsample uses Image.NEAREST for both directions (as in the upstream port),
which is more aggressive than a bilinear/bicubic resize — kept identical on purpose.
"""
import random
from io import BytesIO
from PIL import Image, ImageFilter

LEVEL_LABELS = {0: "clean", 1: "DS0.5", 2: "DS0.25", 3: "JPEG65",
                4: "JPEG30", 5: "Blur3", 6: "Blur5"}


def _compress(img, quality=65):
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def _blur(img, sigma=2):
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _downsample(image, scale_factor=0.5):
    original_size = image.size
    new_size = (max(1, int(original_size[0] * scale_factor)),
                max(1, int(original_size[1] * scale_factor)))
    image = image.resize(new_size, Image.NEAREST)
    image = image.resize(original_size, Image.NEAREST)
    return image


def apply_degradation(image, level):
    """Apply degradation `level` (0-6). Level 0 returns the image unchanged."""
    if level == 1:
        return _downsample(image, scale_factor=0.5)
    if level == 2:
        return _downsample(image, scale_factor=0.25)
    if level == 3:
        return _compress(image, quality=65)
    if level == 4:
        return _compress(image, quality=30)
    if level == 5:
        return _blur(image, sigma=3)
    if level == 6:
        return _blur(image, sigma=5)
    return image


# ── Train-time augmentation ───────────────────────────────────────────────────
# Same three corruption FAMILIES as the test pipeline above (identical operators,
# NEAREST downsampling included), but with continuously sampled parameters that
# span the test levels instead of the seven fixed settings. Each family fires
# independently, in random order, so an image gets 0-3 corruptions.
#
# ⚠️ A model trained with this has seen the test-time corruption families →
# report it with an asterisk, like DNA-Det (see the fairness audit).
_AUG = (
    (0.5, lambda img: _compress(img, quality=random.randint(30, 95))),
    (0.4, lambda img: _blur(img, sigma=random.uniform(0.5, 5.0))),
    (0.4, lambda img: _downsample(img, scale_factor=random.uniform(0.25, 1.0))),
)


def random_degradation(image):
    """Randomly corrupt `image` for training augmentation. Returns a PIL image
    of the same size (JPEG round-trips through a buffer, so the result is
    re-materialised as RGB)."""
    ops = [op for p, op in _AUG if random.random() < p]
    random.shuffle(ops)
    for op in ops:
        image = op(image)
    return image.convert("RGB")


# ── OmniDFA train-time augmentation ──────────────────────────────────────────
# Table 8 of "Few-Shot Synthetic Image Attribution" (OmniDFA, arXiv 2509.25682,
# §E.2), applied in the order the table lists:
#   random JPEG  p=0.5, quality (75, 95)
#   resize       scale (0.5, 2.0)
#   hflip        p=0.5
#   RandAugment  p=0.5, magnitude=9, layers=2, WITHOUT shear/translate
#   gaussian blur p=0.5, sigma (0.1, 2.0)
#   normalize    [0,1]        <-- SKIPPED: CLIPImageProcessor already rescales and
#                                 normalises; doing it here would normalise twice.
#
# The paper excludes shear/translate so the local feature extractor never sees
# padding artifacts from beyond the image border — the same reason applies to us.
#
# Milder than random_degradation on purpose: of the seven test levels only DS0.5
# falls inside these ranges (q75-95 misses JPEG65/30, sigma<=2 misses Blur3/5,
# scale>=0.5 misses DS0.25). Report as `omniaug†`, a weaker asterisk than `aug*`.
_RANDAUG_DROP = ("ShearX", "ShearY", "TranslateX", "TranslateY")
_randaug = None


def _randaugment():
    """torchvision RandAugment minus the four geometric ops. Built once, lazily:
    this module is pure-PIL and is imported by scripts with no torchvision."""
    global _randaug
    if _randaug is None:
        from torchvision.transforms import RandAugment

        class _NoShiftRandAugment(RandAugment):
            def _augmentation_space(self, num_bins, image_size):
                space = super()._augmentation_space(num_bins, image_size)
                return {k: v for k, v in space.items() if k not in _RANDAUG_DROP}

        _randaug = _NoShiftRandAugment(num_ops=2, magnitude=9)
    return _randaug


def _rescale(img, scale):
    """Resize by `scale` at native resolution. Unlike _downsample this does NOT
    restore the original size — the CLIP processor resizes to 224 afterwards, so
    the effect is a change of effective detail (down AND up, scale can exceed 1)."""
    w, h = img.size
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                      Image.BICUBIC)


def omnidfa_augment(image):
    """OmniDFA Table 8 augmentation. PIL in, PIL RGB out. The size CHANGES (the
    resize step is not undone); the CLIP processor normalises it to 224 later."""
    if random.random() < 0.5:
        image = _compress(image, quality=random.randint(75, 95))
    image = _rescale(image, random.uniform(0.5, 2.0))
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = _randaugment()(image.convert("RGB"))
    if random.random() < 0.5:
        image = _blur(image, sigma=random.uniform(0.1, 2.0))
    return image.convert("RGB")


AUG_POLICIES = {"corruption": random_degradation, "omnidfa": omnidfa_augment}


if __name__ == "__main__":
    from PIL import ImageDraw
    img = Image.new("RGB", (256, 200))
    ImageDraw.Draw(img).rectangle([20, 20, 180, 150], fill=(200, 30, 60))
    random.seed(0)
    outs = [random_degradation(img) for _ in range(50)]
    assert all(o.size == img.size and o.mode == "RGB" for o in outs), "size/mode changed"
    changed = sum(o.tobytes() != img.tobytes() for o in outs)
    assert changed > 25, f"augmentation barely fires: {changed}/50 changed"
    assert changed < 50, f"augmentation never leaves an image clean: {changed}/50"
    for lvl in range(7):
        assert apply_degradation(img, lvl).size == img.size

    # OmniDFA policy (needs torchvision — CINECA only).
    space = _randaugment()._augmentation_space(31, (200, 256))
    assert not any(k in space for k in _RANDAUG_DROP), \
        f"shear/translate still in the op space: {sorted(space)}"
    assert len(space) >= 8, f"op space collapsed to {sorted(space)}"
    outs = [omnidfa_augment(img) for _ in range(50)]
    assert all(o.mode == "RGB" for o in outs), "mode changed"
    # The resize step always fires, so every output must differ from the input.
    assert all(o.size != img.size or o.tobytes() != img.tobytes() for o in outs)
    scales = {o.size for o in outs}
    assert len(scales) > 10, f"resize is not sampling: {scales}"
    print(f"ok — {changed}/50 corruption-augmented, 7 test levels intact; "
          f"omnidfa ops={len(space)} (no shear/translate), {len(scales)} distinct sizes")
