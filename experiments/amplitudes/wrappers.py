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
        batchsize, num_particles, num_components = inputs.shape
        inputs = inputs.reshape(batchsize, num_particles * num_components)
        out = self.net(inputs)
        return out

class AmplitudeTransformerWrapper(nn.Module):

    def __init__(self, net, extract_mode="mean", use_momcons=False):
        super().__init__()
        self.net = net
        assert extract_mode in ["global_token", "mean"]
        self.extract_mode = extract_mode
        self.use_momcons = use_momcons

    def forward(self, inputs, type_token):
        batchsize, _, _ = inputs.shape

        # type_token
        type_token = encode_type_token(type_token, batchsize, inputs.device)
        inputs = torch.cat((inputs, type_token), dim=-1)

        # remove one particle if use_momentum_conservation
        if self.use_momcons:
            inputs = inputs[:,:-1,:]

        # global_token (collect information here)
        if self.extract_mode == "global_token":
            global_token = torch.zeros((batchsize, 1, inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype)
            global_token[:,:,0] = 1. # encode something
            inputs = torch.cat((global_token, inputs), dim=1)
        
        outputs = self.net(inputs)
        if self.extract_mode == "global_token":
            amplitudes = outputs[:,0,:]
        elif self.extract_mode == "mean":
            amplitudes = outputs.mean(dim=1)
        
        return amplitudes

class AmplitudeCLSTrWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        batchsize, _, _ = inputs.shape

        type_token = encode_type_token(type_token, batchsize, inputs.device)
        inputs = torch.cat((inputs, type_token), dim=-1)
        outputs = self.net(inputs)

        assert outputs.shape[1] == 1
        amplitudes = outputs[:,0,:]
        return amplitudes

class AmplitudeGAPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, type_token):
        # ignore type token
        batchsize, _, _ = inputs.shape

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

    def __init__(self, net, extract_mode="mean", use_momcons=False, reinsert_type_token=False):
        super().__init__()
        self.net = net
        assert extract_mode in ["mean", "global_token"]
        self.extract_mode = extract_mode
        self.use_momcons = use_momcons

        # reinsert_type_token is processed in the experiment class

    def forward(self, inputs: torch.Tensor, type_token):
        batchsize, _, _ = inputs.shape

        multivector, scalars = self.embed_into_ga(inputs, type_token)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        amplitude = self.extract_from_ga(multivector_outputs, scalar_outputs)

        return amplitude

    def embed_into_ga(self, inputs, type_token):
        batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(2)

        # encode type_token in scalars
        scalars = encode_type_token(type_token, batchsize, inputs.device)

        # remove one particle if use_momentum_conservation
        if self.use_momcons:
            scalars = scalars[:,:-1,:]
            multivector = multivector[:,:-1,:,:]

        # global token
        if self.extract_mode == "global_token":
            global_token_mv = torch.zeros((batchsize, 1, multivector.shape[2], multivector.shape[3]),
                                          dtype=multivector.dtype, device=multivector.device)
            global_token_s = torch.zeros((batchsize, 1, scalars.shape[2]),
                                         dtype=multivector.dtype, device=multivector.device)
            global_token_s[:,:,0] = 1.
            multivector = torch.cat((global_token_mv, multivector), dim=1)
            scalars = torch.cat((global_token_s, scalars), dim=1)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Extract scalars from GA representation
        lorentz_scalars = extract_scalar(multivector)[...,0]
        
        if self.extract_mode == "global_token":
            amplitude = lorentz_scalars[:,0,:]
        elif self.extract_mode == "mean":
            amplitude = lorentz_scalars.mean(dim=1)

        return amplitude
