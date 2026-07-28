"""Self-check for the multi-view (patch / patch+spectrum) attributors.

Runs on CPU, no dataset, no CLIP download:  python -m tests.test_multiview

Covers the things that fail SILENTLY — a wrong grid, a view/label misalignment or
a broken fftshift all still produce a training run that "works" and a full table
of meaningless metrics.
"""
import torch

from data.spectral import spectrum
from patch_attribution.model import patch_views, view_logits


def test_patch_grid():
    # Marker image: pixel (i, j) holds the value i*1000 + j, so every crop is
    # identifiable by its top-left value alone.
    H = W = 224
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    img = (ii * 1000 + jj).float().expand(2, 3, H, W).contiguous()

    for ps in (112, 75):
        views = patch_views(img, patch_size=ps)
        assert views.shape == (2, 10, 3, H, W), views.shape
        assert torch.equal(views[:, 0], img), "view 0 must be the untouched image"

        s = (H - ps) // 2
        offsets = [(i, j) for i in (0, s, 2 * s) for j in (0, s, 2 * s)]
        for k, (top, left) in enumerate(offsets, start=1):
            # bicubic resize keeps the corner value approximately, which is enough
            # to tell the nine crops apart (they differ by thousands).
            got = views[0, k, 0, 0, 0].item()
            want = top * 1000 + left
            assert abs(got - want) < 500, \
                f"patch_size={ps} view {k}: corner {got:.0f}, expected ≈{want}"
        assert 2 * s + ps <= H, f"grid overruns the input for patch_size={ps}"


def test_view_label_alignment():
    """reshape(B*V, D) must pair with labels.repeat_interleave(V), not repeat(V)."""
    B, V, D = 3, 4, 5
    x = torch.arange(B * V * D).float().reshape(B, V, D)
    flat = x.reshape(-1, D)
    labels = torch.tensor([7, 8, 9])
    rep = labels.repeat_interleave(V)
    for i in range(B):
        for v in range(V):
            assert torch.equal(flat[i * V + v], x[i, v])
            assert rep[i * V + v] == labels[i], "view/label misalignment"


def test_view_logits():
    """Planted anchors: every view of a sample points at its own class."""
    K, D = 4, 8
    x_anc = torch.eye(K, D) * 2.0
    # Sample b sits near anchor b, in every one of its 3 views.
    x = torch.stack([(x_anc[b] * 1.5).expand(3, D) for b in range(K)])   # (K, 3, D)
    logits = view_logits(x, x_anc, curv=1.0)
    assert logits.shape == (K, K), logits.shape
    assert torch.equal(logits.argmax(dim=1), torch.arange(K)), logits.argmax(dim=1)
    # A single-view call must agree with the model's own argmin-xi rule.
    from losses.attribution_loss import predict_class
    single = x[:, :1]
    assert torch.equal(view_logits(single, x_anc).argmax(dim=1),
                       predict_class(single.squeeze(1), x_anc))


def test_spectrum():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    x[:, :, 10:40, 10:40] += 3.0
    s = spectrum(x)
    assert s.shape == x.shape and torch.isfinite(s).all()
    assert s.std() > 0.5, "spectrum is nearly constant"
    rolled = spectrum(torch.roll(x, shifts=(7, -3), dims=(-2, -1)))
    assert torch.allclose(s, rolled, atol=1e-4), "fftshift/magnitude broken"


if __name__ == "__main__":
    test_patch_grid()
    test_view_label_alignment()
    test_view_logits()
    test_spectrum()
    print("ok")
