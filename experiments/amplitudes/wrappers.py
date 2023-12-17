# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.amplitudes.preprocessing import preprocess_particles

def encode_type_token(type_token, batchsize, device):
    type_token = torch.tensor(type_token, device=device)
    type_token = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
    type_token = type_token.unsqueeze(0).expand(batchsize, *type_token.shape).float()
    return type_token

class AmplitudeMLPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        # ignore type_token (architecture is not permutation invariant)
        out = self.net(inputs)
        return out

class AmplitudeTransformerWrapper(nn.Module):

    def __init__(self, net, average=False):
        super().__init__()
        self.net = net
        self.average = average

    def forward(self, inputs, type_token):
        batchsize, num_inputs = inputs.shape
        inputs = inputs.reshape(batchsize, num_inputs//4, 4)

        type_token = encode_type_token(type_token, batchsize, inputs.device)
        inputs = torch.cat((inputs, type_token), dim=-1)
        
        outputs = self.net(inputs)
        amplitudes = outputs.mean(dim=1) if self.average else outputs[:,0,:]
        
        return amplitudes

class AmplitudeCLSTrWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        batchsize, num_inputs = inputs.shape
        inputs = inputs.reshape(batchsize, num_inputs//4, 4)

        type_token = encode_type_token(type_token, batchsize, inputs.device)
        inputs = torch.cat((inputs, type_token), dim=-1)
        outputs = self.net(inputs)

        assert outputs.shape[1] == 1
        amplitudes = outputs[:,0,:]
        return amplitudes

class AmplitudeGAMLPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, type_token):
        # ignore type token
        batchsize, num_features = inputs.shape
        inputs = inputs.reshape(batchsize, num_features // 4, 4)

        multivector, scalars = self.embed_into_ga(inputs)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)

        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)

        return amplitude

    def embed_into_ga(self, inputs):
        batchsize, num_channels, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        scalars = torch.zeros((batchsize, 1), device=inputs.device)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)[...,0]

        return outputs


class AmplitudeGATrWrapper(nn.Module):

    def __init__(self, net, average=False):
        super().__init__()
        self.net = net
        self.average = average

    def forward(self, inputs: torch.Tensor, type_token):
        batchsize, num_features = inputs.shape
        inputs = inputs.reshape(batchsize, num_features // 4, 4)

        multivector, scalars = self.embed_into_ga(inputs, type_token)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        mv_outputs = self.extract_from_ga(multivector_outputs, scalar_outputs)

        amplitude = mv_outputs.mean(dim=1) if self.average else mv_outputs[:,0,:]

        return amplitude

    def embed_into_ga(self, inputs, type_token):
        batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(2)

        # encode type_token in scalars
        scalars = encode_type_token(type_token, batchsize, inputs.device)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)[...,0]

        return outputs
