import torch
from geometry.lorentz import exp_map0, log_map0, pairwise_dist

EPS = 1e-5


def test_log_is_inverse_of_exp():
    """log_map0(exp_map0(v)) should recover v."""
    v = torch.randn(8, 128) * 0.1
    x = exp_map0(v)
    v_back = log_map0(x)
    max_diff = (v - v_back).abs().max().item()
    assert max_diff < 1e-3, f"log o exp should be identity, max diff {max_diff}"


def test_distance_from_origin_equals_tangent_norm():
    """For v in tangent space at origin, dist(0, exp_map0(v)) = ||v||."""
    v = torch.randn(8, 128) * 0.1
    x = exp_map0(v)
    origin = torch.zeros(1, 128)
    d = pairwise_dist(origin, x).squeeze(0)
    expected = v.norm(dim=-1)
    assert torch.allclose(d, expected, atol=5e-3), \
        f"got {d} vs {expected}"


def test_zero_maps_to_zero():
    v = torch.zeros(4, 128)
    x = exp_map0(v)
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)


if __name__ == "__main__":
    test_log_is_inverse_of_exp()
    test_distance_from_origin_equals_tangent_norm()
    test_zero_maps_to_zero()
    print("All geometric tests passed.")