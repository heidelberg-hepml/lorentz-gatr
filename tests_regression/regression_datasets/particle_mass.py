# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_vector, extract_scalar
from tests_regression.regression_datasets.constants import DATASET_SIZE, DEVICE


class ParticleMassDataset(torch.utils.data.Dataset):
    """Toy dataset that maps a particle (E, px, py, pz) to its mass m**2 = sqrt(E**2-px**2-py**2-pz**2)."""

    def __init__(self, mass_min=0.01, mass_max=1., p_mu=0., p_std=1.):
        super().__init__()
        self.mass = torch.rand(DATASET_SIZE, 1, 1) * (mass_max - mass_min) + mass_min
        pxyz = torch.randn(DATASET_SIZE, 1, 3) * p_std + p_mu
        E = torch.sqrt(self.mass**2 + torch.sum(pxyz**2, dim=2, keepdims=True))
        self.particle = torch.cat((E, pxyz), dim=2)

        # If there's space on the GPU, let's keep the data on the GPU
        try:
            self.mass.to(DEVICE)
            self.particle.to(DEVICE)
        except RuntimeError:
            pass

    def __len__(self):
        """Return number of samples."""
        return len(self.particle)

    def __getitem__(self, idx):
        """Return datapoint."""
        return self.particle[idx], self.mass[idx]


class ParticleMassWrapper(BaseWrapper):
    """Wrapper around GATr networks for PointsDistanceDataset."""

    mv_in_channels = 1
    mv_out_channels = 1
    s_in_channels = 1
    s_out_channels = 1
    raw_in_channels = 4
    raw_out_channels = 1

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=False)

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector inputs, as expected by geometric network.
        scalar_inputs : torch.Tensor or None
            Scalar inputs, as expected by geometric network.
        """
        batchsize, num_objects, num_features = inputs.shape
        assert num_objects == 1
        assert num_features == 4

        multivector = embed_vector(inputs)  # (batchsize, 1, 16)
        multivector = multivector.unsqueeze(1)  # (batchsize, 1, 1, 16)

        scalars = torch.zeros((batchsize, 1, 1), device=inputs.device)  # (batchsize, 1, 1)
        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from geometric network.
        scalars : torch.Tensor or None
            Scalar outputs from geometric network.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Additional output data, e.g. required for regularization.
        """

        _, num_objects, num_channels, num_ga_components = multivector.shape
        assert num_objects == 1
        assert num_channels == 1
        assert num_ga_components == 16

        norm = extract_scalar(multivector[:, :, 0, :])  # (batchsize, 1, 1)
        return norm, None
