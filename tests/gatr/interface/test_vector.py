import numpy as np
import pytest
import torch

from gatr.interface import embed_vector, extract_vector
from tests.helpers import BATCH_DIMS, TOLERANCES


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
def test_vector_embedding_consistency(batch_dims):
    """Tests whether Lorentz vector embeddings into multivectors are cycle consistent."""
    vectors = torch.randn(*batch_dims, 4)
    multivectors = embed_vector(vectors)
    vectors_reencoded = extract_vector(multivectors)
    torch.testing.assert_close(vectors, vectors_reencoded, **TOLERANCES)
