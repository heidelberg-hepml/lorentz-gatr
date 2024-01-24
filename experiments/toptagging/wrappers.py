# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from xformers.ops.fmha import BlockDiagonalMask

def build_attention_mask(inputs):
    return BlockDiagonalMask.from_seqlens(torch.bincount(inputs.batch).tolist())

class TopTaggingTransformerWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        mask = build_attention_mask(inputs)
        outputs = self.net(inputs.x.unsqueeze(0), attention_mask=mask)
        logits = outputs[0,...][inputs.is_global]

        return logits

class TopTaggingGATrWrapper(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs):
        mask = build_attention_mask(inputs)
        multivector, scalars = self.embed_into_ga(inputs)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars, attention_mask=mask)
        logits = self.extract_from_ga(inputs, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, inputs):

        # encode momenta in multivectors
        multivector = embed_vector(inputs.x)
        multivector = multivector.unsqueeze(1)
        scalars = torch.zeros(len(multivector), 1, device=inputs.x.device, dtype=inputs.x.dtype)
        multivector, scalars = multivector.unsqueeze(0), scalars.unsqueeze(0)

        return multivector, scalars

    def extract_from_ga(self, inputs, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)
        logits = outputs.squeeze().unsqueeze(-1)[inputs.is_global]

        #probs = nn.functional.sigmoid(logits)
        #print(inputs.label.shape, inputs.x.shape, inputs.is_global.shape, probs.shape)
        #print(probs.mean(), probs.std())
        return logits
