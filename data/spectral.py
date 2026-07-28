"""Frequency view of a batch of CLIP-preprocessed images.

Centred log-magnitude FFT, rendered as a 3-channel image the CLIP vision encoder
can ingest directly. Computed on the GPU from the `pixel_values` tensor the
dataloaders already produce — no extra I/O, no new dependency, and the test
images stay byte-identical to the comparison baselines.

FFT rather than DCT on purpose: the centred log-magnitude spectrum is the
standard generator-fingerprint representation (Wang CVPR'20, Frank ICML'20), it
is `torch`-native, and it is "image-like" (radial structure plus fingerprint
spikes) instead of the corner-heavy DCT layout. Note that JPEG's 8x8 block
artefacts live in the frequency domain, so this branch is expected to help the
clean accuracy, not the JPEG robustness.
"""
import torch


def spectrum(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) preprocessed pixels → (B, C, H, W) spectrum image.

    log(1 + |FFT2|), zero frequency shifted to the centre, then standardised per
    image and per channel so the result carries the same statistics CLIP's own
    input normalisation produces.
    """
    mag = torch.log1p(torch.fft.fft2(x.float(), norm="ortho").abs())
    mag = torch.fft.fftshift(mag, dim=(-2, -1))
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True)
    return (mag - mean) / (std + 1e-6)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    x[:, :, 60:160, 60:160] += 3.0        # some structure, not pure noise
    s = spectrum(x)
    assert s.shape == x.shape, s.shape
    assert torch.isfinite(s).all(), "non-finite spectrum"
    assert s.std() > 0.5, "spectrum is nearly constant"
    # |FFT| is invariant to a circular shift of the content: catches a wrong or
    # missing fftshift, which otherwise just silently moves the DC corner around.
    rolled = spectrum(torch.roll(x, shifts=(37, -11), dims=(-2, -1)))
    assert torch.allclose(s, rolled, atol=1e-4), \
        f"not shift-invariant (max diff {(s - rolled).abs().max():.4f})"
    print("ok — spectrum finite, non-constant, shift-invariant")
