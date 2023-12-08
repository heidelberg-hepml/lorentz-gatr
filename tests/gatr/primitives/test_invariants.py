# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Unit tests of bilinear primitives."""

import pytest
import torch

from gatr.primitives import inner_product, squared_norm, abs_squared_norm, pin_invariants
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_invariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_squared_norm_correctness(batch_dims):
    """Checks that squared_norm() is consistent with inner_product()."""
    inputs = torch.randn(*batch_dims, 16)
    squared_norms = squared_norm(inputs)
    true_squared_norms = inner_product(inputs, inputs)
    torch.testing.assert_close(squared_norms, true_squared_norms)


@pytest.mark.parametrize(
    "vector,true_squared_norm",
    [
        ((0.0, 0.0, 0.0, 0.0), 0.0),
        ((1.0, 0.0, 0.0, 0.0), 1.0),
        ((0.0, -2.0, 0.0, 0.0), -4.0),
        ((0.0, 0.0, 3.1, 0.0), -9.61),
        ((0.0, 0.0, 0.0, -1.2), -1.44),
        (None, None),
    ],
)
def test_squared_norm_of_vector(vector, true_squared_norm):
    """Computes the squared norm of a pure vector and compares against a known result."""

    # If vector is None, randomly sample it
    if vector is None:
        vector = torch.randn(4)
        true_squared_norm = vector[0]**2 - torch.sum(vector[1:]**2)

    # Construct multivector
    inputs = torch.zeros(16)
    inputs[1:5] = torch.Tensor(vector)

    # Compute norm
    result = squared_norm(inputs)

    # Validate result
    true_squared_norm = torch.Tensor([true_squared_norm])

    torch.testing.assert_close(result, true_squared_norm)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_inner_product_invariance(batch_dims):
    """Tests the innner_product() primitive for equivariance."""
    check_pin_invariance(inner_product, 2, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_squared_norm_invariance(batch_dims):
    """Tests the squared_norm() primitive for equivariance."""
    check_pin_invariance(squared_norm, 1, batch_dims=batch_dims, **TOLERANCES)

@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_abs_squared_norm_invariance(batch_dims):
    """Tests the abs_squared_norm() primitive for equivariance."""
    check_pin_invariance(abs_squared_norm, 1, batch_dims=batch_dims, **TOLERANCES)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_pin_invariants_invariance(batch_dims):
    """Tests the pin_invariants() primitive for equivariance."""
    check_pin_invariance(pin_invariants, 1, batch_dims=batch_dims, **TOLERANCES)
