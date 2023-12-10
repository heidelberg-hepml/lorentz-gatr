# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from experiments.amplitudes.preprocessing import preprocess_particles

class AmplitudeGATrWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, type_token):
        batchsize, num_features = inputs.shape
        inputs = inputs.reshape(batchsize, num_features // 4, 4)

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
        type_token = torch.tensor(type_token, device=inputs.device)
        scalars = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        scalars = scalars.unsqueeze(0).expand(batchsize, *scalars.shape).to(inputs.dtype)

        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[2:] == (1, 16)
        assert scalars.shape[2:] == (1,)

        # Extract amplitude from one (arbitrary but fixed) object
        amplitude = extract_scalar(multivector[:, 0, 0, :])

        return amplitude


class AmplitudeMLPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        # ignore type_token (architecture is not permutation invariant)
        out = self.net(inputs)
        return out

class AmplitudeTransformerWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token):
        batchsize, num_inputs = inputs.shape
        inputs = inputs.reshape(batchsize, num_inputs//4, 4)

        type_token = torch.tensor(type_token, device=inputs.device)
        type_token = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        type_token = type_token.unsqueeze(0).expand(batchsize, *type_token.shape)
        inputs = torch.cat((inputs, type_token), dim=-1)
        
        outputs = self.net(inputs)
        amplitudes = outputs.mean(dim=1)
        
        return amplitudes
