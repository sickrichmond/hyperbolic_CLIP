"""Self-check for the random pixel blackout of train_attribution.py.

Runs on CPU, no dataset, no CLIP download:  python -m tests.test_blackout

Same failure mode as the mixup check: an augmentation that degenerates to the
identity — or that paints grey instead of black — still trains, still converges,
and only shows up as "the intervention did nothing" 2.5 GPU-hours later.
"""
import torch

# openai/clip-vit-large-patch14; the trainer reads these off train_ds.processor.
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD  = (0.26862954, 0.26130258, 0.27577711)
BLACK = -(torch.tensor(MEAN) / torch.tensor(STD)).view(1, 3, 1, 1)


def _blackout(pixel, blackout_max):
    """The exact expression used in the training loop."""
    lam  = torch.rand(pixel.size(0), 1, 1, 1) * blackout_max
    mask = torch.rand(pixel.size(0), 1, *pixel.shape[-2:]) < lam
    return torch.where(mask, BLACK.to(pixel.dtype), pixel), lam, mask


def _batch(b=8, hw=224):
    return torch.randn(b, 3, hw, hw)


def test_zero_is_identity():
    x = _batch()
    out, _, mask = _blackout(x, 0.0)
    assert not mask.any()
    assert torch.equal(out, x)


def test_masked_fraction_matches_lambda():
    torch.manual_seed(0)
    x = _batch()
    out, lam, mask = _blackout(x, 0.75)
    frac = mask.float().mean(dim=(1, 2, 3))
    # 224² draws per sample → the sampling error is ~1/224 ≈ 0.005.
    assert torch.allclose(frac, lam.view(-1), atol=0.02), (frac, lam.view(-1))
    assert (out != x).any()


def test_lambda_is_per_sample():
    torch.manual_seed(0)
    _, lam, _ = _blackout(_batch(), 0.5)
    assert lam.view(-1).unique().numel() == lam.numel(), "one λ for the whole batch"
    assert (lam <= 0.5).all()


def test_masked_pixels_denormalise_to_black():
    """Not grey: 0 in the NORMALISED tensor would be the CLIP mean colour."""
    torch.manual_seed(0)
    x = _batch(b=4, hw=32)
    out, _, mask = _blackout(x, 0.9)
    raw = out * torch.tensor(STD).view(1, 3, 1, 1) + torch.tensor(MEAN).view(1, 3, 1, 1)
    m = mask.expand_as(raw)
    assert raw[m].abs().max() < 1e-6, raw[m].abs().max()
    # the channels move together: a masked pixel is black in all three
    assert (out[:, 0][mask[:, 0]] == BLACK[0, 0, 0, 0]).all()


if __name__ == "__main__":
    test_zero_is_identity()
    test_masked_fraction_matches_lambda()
    test_lambda_is_per_sample()
    test_masked_pixels_denormalise_to_black()
    print("ok")
