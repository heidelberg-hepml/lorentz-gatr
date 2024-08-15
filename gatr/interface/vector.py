# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def embed_vector(fourvector: torch.Tensor) -> torch.Tensor:
    """Embeds a fourvector tensor into multivectors.

    Parameters
    ----------
    fourvector: torch.Tensor with shape (..., 1)
        Fourvector inputs.

    Returns
    -------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector outputs. `multivectors[..., [0]]` is the same as `fourvector`. The other components
        are zero.
    """

    batch_shape = fourvector.shape[:-1]
    multivector = torch.zeros(
        *batch_shape, 16, dtype=fourvector.dtype, device=fourvector.device
    )
    multivector[..., 0] = fourvector[..., 0]
    multivector[..., 2:5] = fourvector[..., 1:]
    return multivector


def extract_vector(multivectors: torch.Tensor) -> torch.Tensor:
    """Extracts fourvector components from multivectors.

    Parameters
    ----------
    multivectors: torch.Tensor with shape (..., 16)
        Multivector inputs.

    Returns
    -------
    fourvector: torch.Tensor with shape (..., 1)
        Fourvector component of multivectors.
    """

    energy = multivectors[..., [0]]
    momentum = multivectors[..., 2:5]
    fourvector = torch.cat((energy, momentum), dim=-1)

    return fourvector
