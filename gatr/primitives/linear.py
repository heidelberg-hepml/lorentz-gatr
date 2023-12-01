# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache
from pathlib import Path

import torch

from gatr.utils.einsum import cached_einsum, custom_einsum

_FILENAME = "linear_basis.pt"

@lru_cache()
def _load_pin_equi_linear_basis(
    device=torch.device("cpu"), dtype=torch.float32
) -> torch.Tensor:
    """Constructs basis elements for Pin(3,0,1)-equivariant linear maps between multivectors.

    This function is cached.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    basis : torch.Tensor with shape (7, 16, 16)
        Basis elements for equivariant linear maps.
    """

    # To avoid duplicate loading, base everything on float32 CPU version
    if device not in [torch.device("cpu"), "cpu"] and dtype != torch.float32:
        basis = _load_pin_equi_linear_basis()
    else:
        filename = Path(__file__).parent.resolve() / "data" / _FILENAME
        sparse_basis = torch.load(filename).to(torch.float32)
        # Convert to dense tensor
        # The reason we do that is that einsum is not defined for sparse tensors
        basis = sparse_basis.to_dense()

    return basis.to(device=device, dtype=dtype)


@lru_cache()
def _compute_reversal(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector reversal.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    reversal_diag : torch.Tensor with shape (16,)
        The diagonal of the reversal matrix, consisting of +1 and -1 entries.
    """
    reversal_flat = torch.ones(16, device=device, dtype=dtype)
    reversal_flat[5:15] = -1
    return reversal_flat


@lru_cache()
def _compute_grade_involution(device=torch.device("cpu"), dtype=torch.float32) -> torch.Tensor:
    """Constructs a matrix that computes multivector grade involution.

    Parameters
    ----------
    device : torch.device
        Device
    dtype : torch.dtype
        Dtype

    Returns
    -------
    involution_diag : torch.Tensor with shape (16,)
        The diagonal of the involution matrix, consisting of +1 and -1 entries.
    """
    involution_flat = torch.ones(16, device=device, dtype=dtype)
    involution_flat[1:5] = -1
    involution_flat[11:15] = -1
    return involution_flat


NUM_PIN_LINEAR_BASIS_ELEMENTS = len(_load_pin_equi_linear_basis())


def equi_linear(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Pin-equivariant linear map f(x) = sum_{a,j} coeffs_a W^a_ij x_j.

    The W^a are seven pre-defined basis elements.

    Parameters
    ----------
    x : torch.Tensor with shape (..., in_channels, 16)
        Input multivector. Batch dimensions must be broadcastable between x and coeffs.
    coeffs : torch.Tensor with shape (out_channels, in_channels, 7)
        Coefficients for the 59 basis elements. Batch dimensions must be broadcastable between x and
        coeffs.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Result. Batch dimensions are result of broadcasting between x and coeffs.
    """
    basis = _load_pin_equi_linear_basis(device=x.device, dtype=x.dtype)
    return custom_einsum("y x a, a i j, ... x j -> ... y i", coeffs, basis, x, path=[0, 1, 0, 1])


def grade_project(x: torch.Tensor) -> torch.Tensor:
    """Projects an input tensor to the individual grades.

    The return value is a single tensor with a new grade dimension.

    NOTE: this primitive is not used widely in our architectures.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 5, 16)
        Output multivector. The second-to-last dimension indexes the grades.
    """

    # Select kernel on correct device
    basis = _load_pin_equi_linear_basis(device=x.device, dtype=x.dtype, normalize=False)

    # First five basis elements are grade projections
    basis = basis[:5]

    # Project to grades
    projections = cached_einsum("g i j, ... j -> ... g i", basis, x)

    return projections


def reverse(x: torch.Tensor) -> torch.Tensor:
    """Computes the reversal of a multivector.

    The reversal has the same scalar, vector, and pseudoscalar components, but flips sign in the
    bivector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """
    return _compute_reversal(device=x.device, dtype=x.dtype) * x


def grade_involute(x: torch.Tensor) -> torch.Tensor:
    """Computes the grade involution of a multivector.

    The reversal has the same scalar, bivector, and pseudoscalar components, but flips sign in the
    vector and trivector components.

    Parameters
    ----------
    x : torch.Tensor with shape (..., 16)
        Input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 16)
        Output multivector.
    """

    return _compute_grade_involution(device=x.device, dtype=x.dtype) * x
