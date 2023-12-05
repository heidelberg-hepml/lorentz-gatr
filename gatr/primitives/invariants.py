# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache

import torch
import torch.linalg
import math

from gatr.primitives.linear import _compute_reversal, grade_project
from gatr.utils.einsum import cached_einsum

@lru_cache
def _load_inner_product_factors(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        factors = _load_inner_product_factors(kind)
    else:
        _INNER_PRODUCT_FACTORS = [1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
        factors = torch.tensor(_INNER_PRODUCT_FACTORS).to_dense()
    return factors.to(device=device, dtype=dtype)

def inner_product(x: torch.Tensor, y: torch.Tensor, channel_sum: bool = False) -> torch.Tensor:
    """Computes the inner product of multivectors f(x,y) = <x, y> = <~x y>_0.

    In addition to summing over the 16 multivector dimensions, this function also sums
    over an additional channel dimension if channel_sum == True.

    Equal to `geometric_product(reverse(x), y)[..., [0]]` (but faster).

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16) or (..., channels, 16)
        First input multivector. Batch dimensions must be broadcastable between x and y.
    y : torch.Tensor with shape (..., 16) or (..., channels, 16)
        Second input multivector. Batch dimensions must be broadcastable between x and y.
    channel_sum: bool
        Whether to sum over the second-to-last axis (channels)

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Result. Batch dimensions are result of broadcasting between x and y.
    """

    x = x * _load_inner_product_factors()

    if channel_sum:
        outputs = cached_einsum("... c i, ... c i -> ...", x, y)
    else:
        outputs = cached_einsum("... i, ... i -> ...", x, y)

    # We want the output to have shape (..., 1)
    outputs = outputs.unsqueeze(-1)

    return outputs


def squared_norm(x: torch.Tensor) -> torch.Tensor:
    """Computes the GA norm of an input multivector.

    Equal to sqrt(inner_product(x, x)).

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 1)
        Geometric algebra norm of x.
    """

    return inner_product(x, x)


def pin_invariants(x: torch.Tensor) -> torch.Tensor:
    """Computes five invariants from multivectors: scalar component, norms of the four other grades.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5)
        Invariants computed from input multivectors
    """

    # Project to grades
    projections = grade_project(x)  # (..., 5, 16)

    # Compute norms
    squared_norms = inner_product(projections, projections)[..., 0]  # (..., 5)
    norms = torch.sqrt(torch.clamp(squared_norms, 0.0))

    # Outputs: scalar component of input and norms of four other grades
    return torch.cat((x[..., [0]], norms[..., 1:]), dim=-1)  # (..., 5)

@lru_cache()
def ga_metric_grades(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Generate tensor of the diagonal of the GA metric, combined with a grade projection.

    Parameters
    ----------
    device
    dtype

    Returns
    -------
    torch.Tensor of shape [5, 16]
    """
    m = _load_inner_product_factors(device, dtype)
    m_grades = torch.zeros(5, 16, device=device, dtype=dtype)
    offset = 0
    for k in range(4 + 1):
        d = math.comb(4, k)
        m_grades[k, offset : offset + d] = m[offset : offset + d]
        offset += d
    return m_grades

def abs_squared_norm(x: torch.Tensor) -> torch.Tensor:
    m = ga_metric_grades(device=x.device, dtype=x.dtype)
    squared_norms = cached_einsum("... c i, ... c i, g i -> ... c g", x, x, m).abs().sum(-1, keepdim=True)
    return squared_norms
