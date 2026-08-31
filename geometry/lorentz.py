# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from HySAC (https://github.com/aimagelab/HySAC).

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """Pairwise Lorentzian inner product between rows of x and y."""
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Pairwise geodesic distance between two batches of points on the hyperboloid."""
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv ** 0.5


def elementwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """Element-wise Lorentzian inner product between two batches."""
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))
    xyl = torch.sum(x * y, dim=-1) - x_time * y_time
    return xyl


def elementwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Element-wise geodesic distance between corresponding points."""
    c_xyl = -curv * elementwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv ** 0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map a Euclidean tangent vector at the origin onto the hyperboloid.

    Args:
        x: shape (B, D), Euclidean vectors interpreted as tangent vectors
            at the vertex of the hyperboloid.

    Returns:
        Tensor of same shape as `x`, giving space components on the hyperboloid.
    """
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN or Inf detected in input to exp_map0")

    x_norm = torch.norm(x, dim=-1, keepdim=True)
    rc_xnorm = curv ** 0.5 * x_norm

    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2 ** 15))
    rc_xnorm_clamped = torch.clamp(rc_xnorm, min=eps)

    _output = torch.sinh(sinh_input) * x / rc_xnorm_clamped

    if torch.isnan(_output).any() or torch.isinf(_output).any():
        print("NaN or Inf detected in output of exp_map0")

    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5) -> Tensor:
    """Inverse of exp_map0: map points from the hyperboloid to tangent space at origin."""
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x ** 2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv ** 0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-5
) -> Tensor:
    """Half aperture of the entailment cone at point x. Used in step 2."""
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv ** 0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))
    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-5):
    """Exterior angle at x in the hyperbolic triangle Oxy. Used in step 2."""
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1))
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl ** 2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
    return _angle

def axis_ray_dist(
    x: Tensor, a: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """Geodesic distance from each point to the cone AXIS RAY of its anchor.

    The axis of the entailment cone at apex `a` is the geodesic from the origin through
    `a`, and the cone opens along it AWAY from the origin -- so the axis proper is the
    RAY from `a` outward, not the whole geodesic:

        x_par  = <x, a_hat>                 a_hat = a/||a||   (space components)
        x_perp = ||x - x_par * a_hat||

        t* >= r_a   <=>   x_par / x_time >= ||a|| / a_time

          perpendicular branch:  d = asinh(sqrt(c) * x_perp) / sqrt(c)
          apex branch:           d = d_H(a, x)

    The branch test is exact and needs no artanh: both sides are tanh of a radius
    (tanh(sqrt(c) r) = ||.|| / ._time), and tanh is increasing.

    Measuring to the full GEODESIC instead gives sinh(sqrt(c) d) = sqrt(c) x_perp
    unconditionally, which is BILATERAL -- ||x_perp|| is invariant under x -> -x, so
    theta=160 deg scores exactly as theta=20 deg and "descending" past 90 deg means
    walking toward the antipode. That is the degeneracy losses/axis_cone_loss.py
    rejects, and it is why the ray matters. Restricting to the ray makes the result
    strictly monotone in the angle over all of [0, pi]. The ray from the ORIGIN is
    one-sided too, but goes FLAT past 90 deg -- there the nearest point is the origin
    and the anchor drops out of the expression entirely, a dead gradient zone exactly
    where random anchors start. From the APEX the far side is covered by d_H(a, x),
    which keeps a non-zero gradient on both operands everywhere.

    Gradients, and why both branches are needed:
      perpendicular  reads only the anchor's DIRECTION, so it ROTATES the anchor;
      apex           reads the anchor as a point, so it moves the anchor RADIALLY.
    Anchor depth is a learnable quantity only because of the second one. The apex branch
    also pulls an image that is SHALLOWER than its anchor back outward, which is the one
    configuration in which oxy_angle saturates at pi and the cone hinge has no gradient
    at all.

    Both arguments are (B, D) space components, aligned per sample -- pass
    `x_anc[labels]`, not the (K, D) anchor block. Returns (B,).
    """
    rc = curv ** 0.5
    a_norm = torch.norm(a, dim=-1, keepdim=True)                      # (B, 1)
    a_hat = a / a_norm.clamp_min(eps)
    x_par = torch.sum(x * a_hat, dim=-1, keepdim=True)                # (B, 1)

    # From the vector difference, and with eps INSIDE the sqrt. Two separate reasons:
    # sqrt(||x||^2 - x_par^2) cancels catastrophically near the axis, and torch.norm has
    # a NaN gradient at exactly zero -- which is the point this term drives every sample
    # toward, so it is reached in practice rather than in theory.
    diff = x - x_par * a_hat
    x_perp = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps ** 2)      # (B,)

    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))
    a_time = torch.sqrt(1 / curv + a_norm ** 2)

    d_perp = torch.asinh(rc * x_perp) / rc
    d_apex = elementwise_dist(a, x, curv=curv)
    beyond = (x_par / x_time) >= (a_norm / a_time)                    # (B, 1) bool
    return torch.where(beyond.squeeze(-1), d_perp, d_apex)
