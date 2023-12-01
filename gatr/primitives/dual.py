# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache
from typing import Tuple

import torch

@lru_cache()
@torch.no_grad()
def _compute_dualization(
    device=torch.device("cpu"), dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs a tensor for the dual operation.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    permutation : list of int
        Permutation index list to compute the dual
    factors : torch.Tensor
        Signs to multiply the dual outputs with.
    """
    permutation = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    factors = torch.tensor(
        [1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1], device=device, dtype=dtype
    )
    return permutation, factors

def dual(x: torch.Tensor) -> torch.Tensor:
    """Computes the dual of `inputs` (non-equivariant!).

    See Table 4 in the reference.

    References
    ----------
    Leo Dorst, "A Guided Tour to the Plane-Based Geometric Algebra PGA",
        https://geometricalgebra.org/downloads/PGA4CS.pdf

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector, of which we want to compute the dual.

    Returns
    -------
    outputs : torch.Tensor with shale (..., 16)
        The dual of `inputs`, using the pseudoscalar component of `reference` as basis.
    """

    # Select factors on correct device
    perm, factors = _compute_dualization(x.device, x.dtype)

    # Compute dual
    result = factors * x[..., perm]

    return result
