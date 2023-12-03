# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache
import torch

from gatr.primitives.invariants import abs_squared_norm
from gatr.utils.einsum import cached_einsum

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
    abs_squared_norms = abs_squared_norm(x)
    abs_squared_norms = torch.mean(abs_squared_norms, dim=channel_dim, keepdim=True)

    # Insure against low-norm tensors (which can arise even when `x.var(dim=-1)` is high b/c some
    # entries don't contribute to the inner product / GP norm!)
    abs_squared_norms = torch.clamp(abs_squared_norms, epsilon)

    # Rescale inputs
    outputs = gain * x / torch.sqrt(abs_squared_norms)

    return outputs
