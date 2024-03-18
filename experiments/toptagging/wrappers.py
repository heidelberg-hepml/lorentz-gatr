# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import extract_scalar
from xformers.ops.fmha import BlockDiagonalMask

from experiments.logger import LOGGER


def attention_mask(batch, force_xformers=True):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch_geometric.data.Data
        torch_geometric representation of the data
    force_xformers: bool
        Decides whether a xformers or torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
    """
    bincounts = torch.bincount(batch.batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if not force_xformers:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch.batch), len(batch.batch)))
    return mask


class TopTaggingTransformerWrapper(nn.Module):
    """
    Baseline Transformer for top-tagging
    This is mainly for debugging
    """

    def __init__(self, net, force_xformers=True):
        super().__init__()
        self.net = net
        self.force_xformers = force_xformers

    def forward(self, batch):
        mask = attention_mask(batch, self.force_xformers)

        # unsqueeze and squeeze to formally add the batch index (any better way of doing this?)
        inputs = batch.x.unsqueeze(0)
        outputs = self.net(inputs, attention_mask=mask)
        logits = outputs.squeeze(0)[batch.is_global]

        return logits


class TopTaggingGATrWrapper(nn.Module):
    """
    GATr for toptagging
    including all kinds of options to play with
    """

    def __init__(
        self,
        net,
        mean_aggregation=False,
        add_pt=False,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.mean_aggregation = mean_aggregation
        self.add_pt = add_pt
        self.force_xformers = force_xformers

    def forward(self, batch):
        mask = attention_mask(batch, self.force_xformers)

        multivector, scalars = self.embed_into_ga(batch)

        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        # embedding happens in the dataset for convenience
        # add artificial batch index (needed for xformers attention)
        multivector, scalars = batch.x.unsqueeze(0), batch.scalars.unsqueeze(0)
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        outputs = extract_scalar(multivector).squeeze()
        if self.mean_aggregation:
            outputs = outputs.squeeze()
            batchsize = max(batch.batch) + 1
            logits = torch.zeros(batchsize, device=outputs.device, dtype=outputs.dtype)
            logits.index_add_(0, batch.batch, outputs)  # sum
            logits = logits / torch.bincount(batch.batch)  # mean
        else:
            logits = outputs.unsqueeze(-1)[batch.is_global]

        return logits


class QGTaggingGATrWrapper(nn.Module):
    """
    GATr for quark gluon tagging
    including all kinds of options to play with
    """

    def __init__(
        self,
        net,
        mean_aggregation=False,
        add_pt=False,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.mean_aggregation = mean_aggregation
        self.add_pt = add_pt
        self.force_xformers = force_xformers

    def forward(self, batch):
        mask = attention_mask(batch, self.force_xformers)

        multivector, scalars = self.embed_into_ga(batch)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        # embedding happens in the dataset for convenience
        # add artificial batch index (needed for xformers attention)
        multivector, scalars = batch.x.unsqueeze(0), batch.scalars.unsqueeze(0)
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        outputs = extract_scalar(multivector).squeeze()
        if self.mean_aggregation:
            outputs = outputs.squeeze()
            batchsize = max(batch.batch) + 1
            logits = torch.zeros(batchsize, device=outputs.device, dtype=outputs.dtype)
            logits.index_add_(0, batch.batch, outputs)  # sum
            logits = logits / torch.bincount(batch.batch)  # mean
        else:
            logits = outputs.unsqueeze(-1)[batch.is_global]

        return logits
