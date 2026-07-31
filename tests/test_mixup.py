"""Self-check for the tangent-space mixup of train_attribution.py.

Runs on CPU, no dataset, no CLIP download:  python -m tests.test_mixup

The failure mode this guards against is silent: a mixup that degenerates to the
identity (or to a wrong label pairing) still trains, still converges, and only
shows up as "the intervention did nothing" 21 GPU-hours later.
"""
import torch

from geometry.lorentz import exp_map0, log_map0
from losses.attribution_loss import EntailmentConeLoss

CURV = 1.0


def _mix(x, labels, lam, perm):
    """The exact expression used in the training loop."""
    t = log_map0(x.float(), curv=CURV)
    return exp_map0(lam * t + (1 - lam) * t[perm], curv=CURV), labels[perm]


def test_log_exp_roundtrip():
    x = exp_map0(torch.randn(64, 32) * 2, curv=CURV)
    assert torch.allclose(exp_map0(log_map0(x, curv=CURV), curv=CURV), x, atol=1e-4)


def test_lam_one_is_identity():
    """λ=1 must reproduce the un-mixed batch: both the points and the loss."""
    torch.manual_seed(0)
    x = exp_map0(torch.randn(16, 32), curv=CURV)
    anc = exp_map0(torch.randn(4, 32) * 3, curv=CURV)
    labels = torch.randint(0, 4, (16,))
    perm = torch.randperm(16)

    x_mix, _ = _mix(x, labels, 1.0, perm)
    assert torch.allclose(x_mix, x, atol=1e-4)

    loss = EntailmentConeLoss(curv=CURV, min_radius=0.5, margin=0.3)
    l_mix, _ = loss(x_mix, anc, labels)
    l_ref, _ = loss(x, anc, labels)
    assert torch.allclose(l_mix, l_ref, atol=1e-5)


def test_mix_moves_points_and_pairs_labels():
    """λ=0.5 must actually move the points, and label[perm] must follow the same
    permutation used for the embeddings — a mismatch here trains on noise."""
    torch.manual_seed(0)
    x = exp_map0(torch.randn(16, 32) * 2, curv=CURV)
    labels = torch.arange(16)
    perm = torch.randperm(16)
    assert (perm != torch.arange(16)).any(), "degenerate permutation, reseed"

    x_mix, labels_b = _mix(x, labels, 0.5, perm)
    moved = (x_mix - x).norm(dim=-1)
    swapped = perm != torch.arange(16)
    assert (moved[swapped] > 1e-3).all(), "mixup left swapped rows unchanged"
    assert (labels_b == perm).all(), "second target is not the permuted label"


def test_gradient_flows():
    x = exp_map0(torch.randn(8, 32, requires_grad=True), curv=CURV)
    x.retain_grad()
    anc = exp_map0(torch.randn(3, 32) * 3, curv=CURV)
    labels = torch.randint(0, 3, (8,))
    perm = torch.randperm(8)
    loss = EntailmentConeLoss(curv=CURV, min_radius=0.5, margin=0.3)
    x_mix, labels_b = _mix(x, labels, 0.3, perm)
    la, _ = loss(x_mix, anc, labels)
    lb, _ = loss(x_mix, anc, labels_b)
    (0.3 * la + 0.7 * lb).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


if __name__ == "__main__":
    test_log_exp_roundtrip()
    test_lam_one_is_identity()
    test_mix_moves_points_and_pairs_labels()
    test_gradient_flows()
    print("ok")
