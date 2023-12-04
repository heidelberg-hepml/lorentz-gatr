# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from experiments.baselines.transformer import BaselineTransformer
from experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_vector, extract_scalar
from experiments.amplitudes.preprocessing import preprocess_particles

class AmplitudeGATrWrapper(BaseWrapper):
    """Wraps around GATr for the n-body prediction experiment.

    Parameters
    ----------
    net : torch.nn.Module
        GATr model that accepts inputs with 1 multivector channel and 1 scalar channel, and
        returns outputs with 1 multivector channel and 1 scalar channel.
    """

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=False)
        self.supports_variable_items = True

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        Parameters
        ----------
        inputs : torch.Tensor with shape (batchsize, objects, 7)
            n-body initial state: a concatenation of masses, initial positions, and initial
            velocities along the feature dimension.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector representation of masses, positions, and velocities.
        scalar_inputs : torch.Tensor or None
            Dummy auxiliary scalars, containing no information.
        """
        batchsize, num_objects, _ = inputs.shape

        # Build one multivector holding masses, positions, and velocities for each object
        multivector = embed_vector(inputs)

        # Scalar inputs are not really needed here
        scalars = torch.zeros((batchsize, num_objects, 1), device=inputs.device)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        """Extracts raw outputs from the GATr multivector + scalar outputs.

        We parameterize the predicted final positions as points.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from GATr.
        scalars : torch.Tensor or None
            Scalar outputs from GATr.

        Returns
        -------
        outputs : torch.Tensor
            Predicted final-state positions.
        other : torch.Tensor
            Regularization terms.
        """

        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[2:] == (1, 16)
        assert scalars.shape[2:] == (1,)

        # Extract position
        amplitude = extract_scalar(multivector[:, :, 0, :])

        return amplitude


class AmplitudeBaselineWrapper(nn.Module):
    """Wraps around simple baselines (MLP or Transformer) for the amplitude regression experiment.

    Parameters
    ----------
    net : torch.nn.Module
        Model that accepts inputs with 1 channel and returns outputs with 1 channel
    """

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.supports_variable_items = isinstance(
            net, (BaselineTransformer)
        )
        self.mean, self.std = torch.tensor(mean).float(), torch.tensor(std).float()

    def forward(self, inputs):
        """Wrapped forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        other : torch.Tensor
            Dummy term, since the baselines do not require regularization.
        """
        inputs, _, _ = preprocess_particles(inputs, mean=self.mean, std=self.std)
        
        batchsize = inputs.shape[0]
        return self.net(inputs)
