"""Self-check for the image-centroid anchors + train-time augmentation.

Runs on CPU, no dataset needed:  python -m tests.test_centroid_anchors

Covers the two things that fail silently:
  1. the anchor REORDERING from the checkpoint's class order ('real' first) to
     the harness label order ('real' last) — a wrong permutation still produces
     a full set of metrics, just meaningless ones;
  2. random_degradation actually firing (and never changing the image size,
     which would break the batch collation).
"""
import os

os.environ.setdefault("IAB_EXCLUDE_GENERATORS", "dalle3")   # 22-class map

import torch
from PIL import Image, ImageDraw

from comparison.training.test_hypclip import harness_class_names, load_anchors
from data.degradations import random_degradation


def test_anchor_reordering():
    harness = harness_class_names()
    assert harness[-1] == "real", f"expected 'real' last in harness order, got {harness[-1]}"
    K = len(harness)
    assert K == 22, f"IAB_EXCLUDE_GENERATORS=dalle3 should give 22 classes, got {K}"

    # Checkpoint order = train_attribution.build_anchors(): 'real' first.
    ckpt_names = ["real"] + [n for n in harness if n != "real"]
    # Row i is the i-th scaled basis vector → the class it belongs to is recoverable
    # from argmax, so a wrong permutation cannot pass.
    tangent = torch.eye(K) * 2.0
    ckpt = {"class_names": ckpt_names, "anchor_tangent": tangent}

    x_anc = load_anchors(ckpt, model=None, curv=1.0, device="cpu")
    assert x_anc.shape == (K, K), x_anc.shape
    for i, name in enumerate(harness):
        expected_row = ckpt_names.index(name)
        got = int(x_anc[i].argmax())
        assert got == expected_row, f"class {name}: anchor row {got}, expected {expected_row}"
    # exp_map0 only rescales along the direction — norms must stay equal and > 0
    norms = x_anc.norm(dim=-1)
    assert torch.allclose(norms, norms[0]) and norms[0] > 0, norms

    missing = {"class_names": ckpt_names[:-1], "anchor_tangent": tangent[:-1]}
    try:
        load_anchors(missing, model=None, curv=1.0, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("a checkpoint missing a class must not load silently")


def test_random_degradation():
    img = Image.new("RGB", (256, 200))
    ImageDraw.Draw(img).rectangle([20, 20, 180, 150], fill=(200, 30, 60))
    outs = [random_degradation(img) for _ in range(50)]
    assert all(o.size == img.size and o.mode == "RGB" for o in outs)
    changed = sum(o.tobytes() != img.tobytes() for o in outs)
    assert 25 < changed < 50, f"{changed}/50 augmented — check the per-family probabilities"


if __name__ == "__main__":
    test_anchor_reordering()
    test_random_degradation()
    print("ok")
