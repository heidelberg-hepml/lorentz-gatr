# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar

class TopTaggingTransformerWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, attention_mask):
        batchsize, _, _ = inputs.shape

        # global_token (collect information here)
        global_token = torch.zeros((batchsize, 1, inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype)
        global_token[:,:,0] = 1. # encode something
        inputs = torch.cat((global_token, inputs), dim=1)

        outputs = self.net(inputs, attention_mask=attention_mask)
        logits = outputs[:,0,:]

        return logits

class TopTaggingGATrWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor, attention_mask):
        batchsize, _, _ = inputs.shape

        multivector, scalars = self.embed_into_ga(inputs)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars, attention_mask=attention_mask)
        probs = self.extract_from_ga(multivector_outputs, scalar_outputs)

        return probs

    def embed_into_ga(self, inputs):
        batchsize, num_objects, _ = inputs.shape

        # encode momenta in multivectors
        multivector = embed_vector(inputs)
        multivector = multivector.unsqueeze(2)
        scalars = torch.zeros(batchsize, num_objects, 1)

        # global token
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
        logits = lorentz_scalars[:,0,:]

        probs = nn.functional.sigmoid(logits)
        return probs
