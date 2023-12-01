# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
import math

from gatr.primitives.invariants import _load_inner_product_factors

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

def abs_squared_norm(self, x: Tensor) -> Tensor:
    m = ga_metric_grades(device=x.device, dtype=x.dtype)
    squared_norms = cached_einsum("... c i, ... c i, g i -> ... c g", x, x, m).abs().sum(-1, keepdim=True)
    return squared_norms

def equi_layer_norm(
    x: torch.Tensor, channel_dim: int = -2, gain: float = 1.0, epsilon: float = 0.01
) -> torch.Tensor:
    """Modified version for the EquiLayerNorm for geometric algebras with negative-norm states,
    following Appendix E in https://arxiv.org/pdf/2311.04744.pdf

    Two aspects are used to stabilize the normalization
    1) Take the absolute value of the scalar products. This avoids negative values in the square root
    2) Replace the absolute value of the scalar product with the sum over absolute values of scalar products
        between graded vectors, with the sum running over all possible grades. This also avoids that the
        contributions of two states to the norm can cancel.

    Parameters
    ----------
    x : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Input multivectors.
    channel_dim : int
        Channel dimension index. Defaults to the second-last entry (last are the multivector
        components).
    gain : float
        Target output scale.
    epsilon : float
        Small numerical factor to avoid instabilities. By default, we use a reasonably large number
        to balance issues that arise from some multivector components not contributing to the norm.

    Returns
    -------
    outputs : torch.Tensor with shape `(batch_dim, *channel_dims, 16)`
        Normalized inputs.
    """

    # Compute mean_channels |inputs|^2
    abs_squared_norms = abs_squared_norm
    abs_squared_norms = torch.mean(abs_squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    abs_squared_norms = torch.clamp(abs_squared_norms, epsilon)

    # Rescale inputs
    outputs = gain * x / torch.sqrt(abs_squared_norms)

    return outputs
