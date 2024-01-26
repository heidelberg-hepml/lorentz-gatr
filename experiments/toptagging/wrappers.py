# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
#import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from xformers.ops.fmha import BlockDiagonalMask

def xformers_mask(batch):
    bincounts = torch.bincount(batch.batch).tolist()
    return BlockDiagonalMask.from_seqlens(bincounts)

def torch_mask(batch):
    bincounts = torch.bincount(batch.batch).tolist()
    blocks = [torch.ones(i, i, dtype=torch.bool, device=batch.x.device) for i in bincounts]
    return torch.block_diag(*blocks)

class TopTaggingTransformerWrapper(nn.Module):

    def __init__(self, net, force_xformers=True):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers

    def forward(self, batch):
        if self.force_xformers:
            mask = xformers_mask(batch)
        else:
            mask = torch_mask(batch)
            
        outputs = self.net(batch.x.unsqueeze(0), attention_mask=mask)
        logits = outputs.squeeze(0)[batch.is_global]

        return logits

class TopTaggingGATrWrapper(nn.Module):

    def __init__(self, net, force_xformers=True):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers

    def forward(self, batch):
        if self.force_xformers:
            mask = xformers_mask(batch)
        else:
            mask = torch_mask(batch)
            
        multivector, scalars = self.embed_into_ga(batch)

        multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars, attention_mask=mask)
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):

        # encode momenta in multivectors
        multivector = embed_vector(batch.x)
        multivector = multivector.unsqueeze(1)
        
        scalars = torch.zeros(multivector.shape[0], 1, device=batch.x.device, dtype=batch.x.dtype)

        multivector, scalars = multivector.unsqueeze(0), scalars.unsqueeze(0)
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        # Extract scalars from GA representation
        outputs = extract_scalar(multivector)
        logits = outputs.squeeze([0,-1])[batch.is_global]

        return logits
