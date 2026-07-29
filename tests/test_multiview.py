"""Self-check for the multi-view (patch / patch+spectrum) attributors.

Runs on CPU, no dataset, no CLIP download:  python -m tests.test_multiview

Covers the things that fail SILENTLY — a wrong grid, a view/label misalignment or
a broken fftshift all still produce a training run that "works" and a full table
of meaningless metrics.
"""
import torch

from data.spectral import spectrum
from patch_attribution.model import PatchAttributionCLIP, patch_views, view_logits


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


class _StubBackbone:
    """encode_views without CLIP: the embedding of a view is its first 4 pixels."""
    patch_size = 112

    def encode_image(self, x):
        return x.flatten(1)[:, :4], None


def test_view_source_dispatch():
    """4-D input → the model cuts the grid; 5-D input → the dataset already did.

    If the 5-D branch were missing, a native-grid batch would be re-cropped and
    silently turn 10 full-resolution views into 100 nonsense ones.
    """
    stub = _StubBackbone()
    out4 = PatchAttributionCLIP.encode_views(stub, torch.randn(2, 3, 224, 224))
    assert out4.shape == (2, 10, 4), out4.shape

    pre = torch.randn(2, 10, 3, 224, 224)
    out5 = PatchAttributionCLIP.encode_views(stub, pre)
    assert out5.shape == (2, 10, 4), out5.shape
    assert torch.equal(out5, pre.flatten(0, 1).flatten(1)[:, :4].view(2, 10, 4)), \
        "5-D input must be used as-is, not re-cropped"
    # a different view count must pass through untouched too
    assert PatchAttributionCLIP.encode_views(stub, torch.randn(2, 3, 3, 224, 224)
                                             ).shape == (2, 3, 4)


def test_native_grid_geometry():
    """The PIL-side grid must match the tensor-side one: same offsets, same
    row-major order, whole image first."""
    from PIL import Image
    from data.iab_clip_dataset import native_patch_grid

    # Non-square on purpose, and every pixel encodes its own coordinates, so each
    # window is identified by its corner alone.
    W, H = 64, 48
    img = Image.new("RGB", (W, H))
    for y in range(H):
        for x in range(W):
            img.putpixel((x, y), (x, y, 0))

    views = native_patch_grid(img)
    assert len(views) == 10, len(views)
    assert views[0].size == (W, H) and views[0].getpixel((0, 0)) == (0, 0, 0), \
        "view 0 must be the whole image"

    pw, ph = W // 2, H // 2
    expected = [(x, y) for y in (0, ph // 2, ph) for x in (0, pw // 2, pw)]
    for k, (ex, ey) in enumerate(expected, start=1):
        assert views[k].size == (pw, ph), views[k].size
        assert views[k].getpixel((0, 0)) == (ex, ey, 0), \
            f"window {k} starts at {views[k].getpixel((0, 0))[:2]}, expected {(ex, ey)}"
        assert ex + pw <= W and ey + ph <= H, "window falls outside the image"


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
    test_view_source_dispatch()
    test_native_grid_geometry()
    test_view_label_alignment()
    test_view_logits()
    test_spectrum()
    print("ok")
