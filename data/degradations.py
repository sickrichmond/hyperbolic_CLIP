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
    print(f"ok — {changed}/50 augmented, 7 test levels intact")
