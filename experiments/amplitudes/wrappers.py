# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from experiments.base_wrapper import BaseWrapper
from gatr.interface import embed_vector, extract_scalar
from experiments.amplitudes.preprocessing import preprocess_particles

class AmplitudeGATrWrapper(BaseWrapper):

    def __init__(self, net):
        super().__init__(net, scalars=True, return_other=False)

    def forward(self, inputs: torch.Tensor, type_token=None):
        batch_size, num_features = inputs.shape
        inputs = inputs.reshape(batch_size, num_features // 4, 4)

        multivector, _ = self.embed_into_ga(inputs)

        type_token = torch.tensor(type_token, device=inputs.device)
        scalars = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        scalars = scalars.unsqueeze(0).expand(batch_size, *scalars.shape).to(inputs.dtype)
        
        multivector = multivector.unsqueeze(2)
        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)
        outputs = self.extract_from_ga(multivector_outputs, scalar_outputs)

        outputs = outputs[...,0,:]
        return outputs

    def embed_into_ga(self, inputs):
        batchsize, num_objects, _ = inputs.shape
        
        # Build one multivector holding masses, positions, and velocities for each object
        multivector = embed_vector(inputs)

        return multivector, None

    def extract_from_ga(self, multivector, scalars):
        # Check channels of inputs. Batchsize and object numbers are free.
        assert multivector.shape[2:] == (1, 16)
        assert scalars.shape[2:] == (1,)

        # Extract position
        amplitude = extract_scalar(multivector[:, :, 0, :])

        return amplitude


class AmplitudeMLPWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token=None):
        # ignore type_token, because architecture is not permutation invariant
        out = self.net(inputs)
        return out

class AmplitudeTransformerWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, type_token=None):
        batch_size, num_inputs = inputs.shape
        inputs = inputs.reshape(batch_size, num_inputs//4, 4)

        type_token = torch.tensor(type_token, device=inputs.device)
        type_token = nn.functional.one_hot(type_token, num_classes=type_token.max()+1)
        type_token = type_token.unsqueeze(0).expand(batch_size, *type_token.shape)
        inputs = torch.cat((inputs, type_token), dim=-1)
        outputs = self.net(inputs)
        amplitudes = outputs.mean(dim=1)
        
        return amplitudes
