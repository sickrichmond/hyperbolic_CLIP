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