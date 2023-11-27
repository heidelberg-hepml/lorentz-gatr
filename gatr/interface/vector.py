"""Functions that embed 4-vectors in the geometric algebra."""


import torch


def embed_vector(vector: torch.Tensor) -> torch.Tensor:
    """Embeds 4-vectors in multivectors.

    Parameters
    ----------
    coordinates : torch.Tensor with shape (..., 4)
        4-vector coordinates

    Returns
    -------
    multivector : torch.Tensor with shape (..., 16)
        Embedding into multivector.
    """

    # Create multivector tensor with same batch shape, same device, same dtype as input
    batch_shape = vector.shape[:-1]
    multivector = torch.zeros(*batch_shape, 16, dtype=vector.dtype, device=vector.device)

    # Embedding into 4-vectors
    multivector[..., 1:5] = vector

    return multivector


def extract_point(multivector: torch.Tensor) -> torch.Tensor:
    """Given a multivector, extract a 4-vector from the vector components.

    Parameters
    ----------
    multivector : torch.Tensor with shape (..., 16)
        Multivector.

    Returns
    -------
    vector : torch.Tensor with shape (..., 4)
        4-vector corresponding to the vector components of the multivector.
    """

    vector = torch.cat(multivector[..., 1:5], dim=-1)

    return vector
