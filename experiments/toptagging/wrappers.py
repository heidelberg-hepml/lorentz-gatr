# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
from xformers.ops.fmha import BlockDiagonalMask

from experiments.logger import LOGGER


def xformers_mask(batch):
    # efficient implementation, should use this for serious business
    bincounts = torch.bincount(batch.batch).tolist()
    return BlockDiagonalMask.from_seqlens(bincounts)


def torch_mask(batch):
    # inefficient implementation due to quadratic mask for very simple task
    # could be implemented more efficiently by masking queries, keys and values before
    # combining them in self-attention, this requires only a linear mask

    # implemented only to allow debugging on cpu's
    bincounts = torch.bincount(batch.batch).tolist()
    blocks = [
        torch.ones(i, i, dtype=torch.bool, device=batch.x.device) for i in bincounts
    ]
    return torch.block_diag(*blocks)


class TopTaggingTransformerWrapper(nn.Module):
    """
    Baseline Transformer for top-tagging
    This is mainly for debugging. I will not all features that we try for GATr
    """

    def __init__(self, net, force_xformers=True):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers

    def forward(self, batch):
        if self.force_xformers:
            mask = xformers_mask(batch)
        else:
            mask = torch_mask(batch)

        # unsqueeze and squeeze to formally add the batch index (any better way of doing this?)
        inputs = batch.x.unsqueeze(0)
        outputs = self.net(inputs, attention_mask=mask)
        logits = outputs.squeeze(0)[batch.is_global]

        return logits


class TopTaggingGATrWrapper(nn.Module):
    def __init__(
        self, net, beam_reference, mean_aggregation=False, force_xformers=True
    ):
        super().__init__()
        self.net = net
        self.beam_reference = beam_reference
        self.mean_aggregation = mean_aggregation
        self.force_xformers = force_xformers
        assert self.beam_reference in [
            None,
            "photon",
            "spacelike",
            "xyplane",
        ], f"beam_reference {self.beam_reference} not implemented"

    def forward(self, batch):
        if self.force_xformers:
            mask = xformers_mask(batch)
        else:
            mask = torch_mask(batch)

        multivector, scalars = self.embed_into_ga(batch)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        # unsqueeze channel dimension (already done in case data.add_pairs=true)
        if len(batch.x.shape) == 2:
            batch.x = batch.x.unsqueeze(1)

        # add beam_reference
        if self.beam_reference in ["photon", "spacelike"]:
            # add another 4-momentum
            beam = [1, 0, 0, 1] if self.beam_reference == "photon" else [0, 0, 0, 1]
            beam = torch.tensor(beam, device=batch.x.device, dtype=batch.x.dtype)
            beam = beam.unsqueeze(0).expand(batch.x.shape[0], 4).unsqueeze(1)

            batch.x = torch.concatenate((batch.x, beam), dim=1)
            multivector = embed_vector(batch.x)
        elif self.beam_reference == "xyplane":
            multivector = embed_vector(batch.x)

            # add the x-y-plane, embedded as a bivector
            # convention for bivector components: [tx, ty, tz, xy, xz, yz]
            plane = torch.zeros(
                multivector.shape[0], 1, 16, device=batch.x.device, dtype=batch.x.dtype
            )
            plane[..., 8] = 1
            multivector = torch.concatenate((multivector, plane), dim=-2)
        else:
            multivector = embed_vector(batch.x)

        scalars = torch.zeros(
            multivector.shape[0], 1, device=batch.x.device, dtype=batch.x.dtype
        )

        # add batch index (needed for xformers attention)
        multivector, scalars = multivector.unsqueeze(0), scalars.unsqueeze(0)
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):

        # remove batch index (0) and channel index (-1)
        outputs = extract_scalar(multivector)
        if self.mean_aggregation:
            outputs = (
                outputs.squeeze()
            )  # remove irrelevant dimensions (batchsize=1, channels=1, mv_dimension=1)
            batch_masks = [batch.batch == n for n in range(batch.batch.max() + 1)]
            logits = torch.cat(
                [outputs[mask].mean(dim=0, keepdim=True) for mask in batch_masks]
            )
        else:
            logits = outputs.squeeze([0, -1])[batch.is_global]

        return logits
