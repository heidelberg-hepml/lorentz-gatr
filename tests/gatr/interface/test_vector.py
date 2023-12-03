# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import pytest
import torch

from gatr.interface import embed_vector, extract_vector
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_vector_embedding_consistency(batch_dims):
    """Tests whether vector embeddings into multivectors are cycle consistent."""
    vector = torch.randn(*batch_dims, 4)
    multivectors = embed_vector(vector)
    vector_reencoded = extract_vector(multivectors)
    torch.testing.assert_close(vector, vector_reencoded, **TOLERANCES)
