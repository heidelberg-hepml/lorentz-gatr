# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
from typing import List

import numpy as np
import torch
from torch import nn

def sum_of_numbers_up_to(n):
    return (n * (n + 1)) // 2

class MLP(nn.Module):
    """A simple baseline MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(self, in_shape, out_shape, hidden_channels, hidden_layers,
                 dropout_prob=None):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = in_shape
        self.out_shape = out_shape

        layers: List[nn.Module] = [nn.Linear(np.product(in_shape), hidden_channels)]
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        for _ in range(hidden_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_channels, np.product(self.out_shape)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of baseline MLP."""
        return self.mlp(inputs)

class DSMLP(nn.Module):
    """
    A modification of the MLP to apply a learnable preprocessing on the inputs based on the Deep Sets framework.
    Once the latent space vectors are generated for each particle, a sequence of momentum invariants is concatenated to them and fed to the MLP.
    """

    def __init__(self, in_shape, out_shape, num_particles_boson, num_particles_glu, hidden_channels_prenet, hidden_layers_prenet, out_dim_prenet_sep, hidden_channels_net, hidden_layers_net):
        super().__init__()
        
        if not hidden_layers_prenet > 0 or not hidden_layers_net > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.num_particles_boson = num_particles_boson
        self.num_particles_glu = num_particles_glu
        self.out_dim_prenet_sep = out_dim_prenet_sep
        
        self.input_dim_contracted = sum_of_numbers_up_to(2+num_particles_boson+num_particles_glu-1)
        
        self.prenet_ini = MLP(
                in_shape=4,
                out_shape=out_dim_prenet_sep, # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
                hidden_channels=hidden_channels_prenet,
                hidden_layers=hidden_layers_prenet,
                dropout_prob=None
        )

        self.prenet_boson = MLP(
                in_shape=4,
                out_shape=out_dim_prenet_sep, # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
                hidden_channels=hidden_channels_prenet,
                hidden_layers=hidden_layers_prenet,
                dropout_prob=None
        )

        self.prenet_jet = MLP(
                in_shape=4,
                out_shape=out_dim_prenet_sep, # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
                hidden_channels=hidden_channels_prenet,
                hidden_layers=hidden_layers_prenet,
                dropout_prob=None
        )


        self.net = MLP(
                in_shape=out_dim_prenet_sep*(2+num_particles_boson+1)+self.input_dim_contracted,
                out_shape=1,
                hidden_channels=hidden_channels_net,
                hidden_layers=hidden_layers_net,
                dropout_prob=None
        )

    def forward(self, x):

        res = torch.cat((self.prenet_ini(x[:,0:4]),self.prenet_ini(x[:,4:8])),1)

        for i in range(self.num_particles_boson):
          res = torch.cat((res, self.prenet_boson(x[:,4*(i+2):4*(i+2+1)])),1)
        
        res = torch.cat((res, sum(self.prenet_jet(x[:,4*(i+2+self.num_particles_boson):4*(i+2+self.num_particles_boson+1)]) for i in range(self.num_particles_glu))), 1)

        res_cont = torch.cat((res,x[:,4*(2+self.num_particles_boson+self.num_particles_glu):]),1)

        return self.net(res_cont)
