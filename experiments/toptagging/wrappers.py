# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_scalar
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


def embed_beam_reference(p_ref, beam_reference):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    p_ref: torch.tensor with shape (..., items, 4)
        Reference tensor to infer device, dtype and shape for the beam_reference
    beam_reference: str
        Different options for adding a beam_reference

    Returns
    -------
    beam: torch.tensor with shape (..., items, mv_channels, 16)
        beam embedded as mv_channels multivectors
    """
    if beam_reference in ["timelike", "spacelike"]:
        # add another 4-momentum
        beam = [1, 0, 0, 1] if beam_reference == "timelike" else [0, 0, 0, 1]
        beam = torch.tensor(beam, device=p_ref.device, dtype=p_ref.dtype)
        beam = beam.unsqueeze(0).expand(p_ref.shape[0], 1, 4)
        beam = embed_vector(beam)

    elif beam_reference == "cgenn":
        beam_mass = 1.0
        beam = [[(1 + beam_mass) ** 0.5, 0, 0, 1], [(1 + beam_mass) ** 0.5, 0, 0, -1]]
        beam = torch.tensor(beam, device=p_ref.device, dtype=p_ref.dtype)
        beam = beam.unsqueeze(0).expand(p_ref.shape[0], 2, 4)
        beam = embed_vector(beam)

    elif beam_reference == "xyplane":
        # add the x-y-plane, embedded as a bivector
        # convention for bivector components: [tx, ty, tz, xy, xz, yz]
        beam = torch.zeros(
            p_ref.shape[0], 1, 16, device=p_ref.device, dtype=p_ref.dtype
        )
        beam[..., 8] = 1

    elif beam_reference is None:
        beam = None

    else:
        raise NotImplementedError

    return beam


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
        beam_reference=None,
        mean_aggregation=False,
        add_pt=False,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.beam_reference = beam_reference
        self.mean_aggregation = mean_aggregation
        self.add_pt = add_pt
        self.force_xformers = force_xformers
        assert self.beam_reference in [
            None,
            "timelike",
            "spacelike",
            "xyplane",
            "cgenn",
        ], f"beam_reference {self.beam_reference} not implemented"

    def forward(self, batch):
        mask = attention_mask(batch, self.force_xformers)

        multivector, scalars = self.embed_into_ga(batch)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        multivector = embed_vector(batch.x)

        beam = embed_beam_reference(multivector, self.beam_reference)
        if beam is not None:
            multivector = torch.cat((multivector, beam), dim=-2)

        # add artificial batch index (needed for xformers attention)
        multivector, scalars = multivector.unsqueeze(0), batch.scalars.unsqueeze(0)
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
