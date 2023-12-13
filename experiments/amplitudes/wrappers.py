# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from experiments.amplitudes.preprocessing import preprocess_particles

class AmplitudeGATrWrapper(nn.Module):

    def __init__(self, net,
                 mlp_blocks, mlp_channels, average):
        super().__init__()
        self.net = net
        self.average = average

        layers = [nn.GELU()] # start with activation, because the transformer finished with a linear layer
        for _ in range(mlp_blocks):
            layers.append(nn.Linear(mlp_channels, mlp_channels))
            layers.append(nn.GELU())
        layers.append(nn.Linear(mlp_channels, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, type_token):
        batchsize, num_features = inputs.shape
        inputs = inputs.reshape(batchsize, num_features // 4, 4)

        multivector, scalars = self.embed_into_ga(inputs, type_token)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        mv_outputs = self.extract_from_ga(multivector_outputs, scalar_outputs)

        outputs = mv_outputs
        #outputs = scalar_outputs
        if self.average:
            outputs = outputs.mean(dim=1)
        else:
            outputs = outputs[:,0,:]
        amplitude = self.mlp(outputs)

        return amplitude

    def embed_into_ga(self, inputs, type_token):
        batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(2)

        # encode type_token in scalars
        type_token = torch.tensor(type_token, device=inputs.device)
        scalars = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        scalars = scalars.unsqueeze(0).expand(batchsize, *scalars.shape).to(inputs.dtype)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)[...,0]

        return outputs


class AmplitudeMLPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        # ignore type_token (architecture is not permutation invariant)
        out = self.net(inputs)
        return out

class AmplitudeTransformerWrapper(nn.Module):

    def __init__(self, net,
                 mlp_blocks, mlp_channels, average):
        super().__init__()
        self.net = net
        self.average = average

        layers = [nn.GELU()] # start with activation, because the transformer finished with a linear layer
        for _ in range(mlp_blocks):
            layers.append(nn.Linear(mlp_channels, mlp_channels))
            layers.append(nn.GELU())
        layers.append(nn.Linear(mlp_channels, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs, type_token):
        batchsize, num_inputs = inputs.shape
        inputs = inputs.reshape(batchsize, num_inputs//4, 4)

        type_token = torch.tensor(type_token, device=inputs.device)
        type_token = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        type_token = type_token.unsqueeze(0).expand(batchsize, *type_token.shape)
        inputs = torch.cat((inputs, type_token), dim=-1)
        
        outputs = self.net(inputs)
        if self.average:
            outputs = outputs.mean(dim=1) # average over transformer set elements
        else:
            outputs = outputs[:,0,:] # pick first transformer set element
        amplitudes = self.mlp(outputs)
        
        return amplitudes
