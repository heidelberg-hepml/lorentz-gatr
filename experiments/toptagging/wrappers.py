import numpy as np
import torch
from torch import nn

from gatr.interface import extract_scalar
from xformers.ops.fmha import BlockDiagonalMask


def xformers_sa_mask(batch, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements

    Parameters
    ----------
    batch: torch_geometric.data.Data
        torch_geometric representation of the data
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    bincounts = torch.bincount(batch.batch).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch.batch), len(batch.batch))).to(
            batch.batch.device
        )
    return mask


def xformers_ca_mask(batch, num_class_tokens=1, materialize=False):
    """
    Construct attention mask that makes sure that objects only attend to each other
    within the same batch element, and not across batch elements
    This version is for class-attention, where the blocks are non-square

    Parameters
    ----------
    batch: torch_geometric.data.Data
        torch_geometric representation of the data
    num_class_tokens: int
        Number of class tokens to be used in class attention
    materialize: bool
        Decides whether a xformers or ('materialized') torch.tensor mask should be returned
        The xformers mask allows to use the optimized xformers attention kernel, but only runs on gpu

    Returns
    -------
    mask: xformers.ops.fmha.attn_bias.BlockDiagonalMask or torch.tensor
        attention mask, to be used in xformers.ops.memory_efficient_attention
        or torch.nn.functional.scaled_dot_product_attention
    """
    kv_seqlen = torch.bincount(batch.batch).tolist()
    kv_seqlen = [i + num_class_tokens for i in kv_seqlen]
    q_seqlen = [num_class_tokens] * len(kv_seqlen)
    mask = BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen=kv_seqlen)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(sum(q_seqlen), sum(kv_seqlen))).to(
            batch.batch.device
        )
    return mask


class TopTaggingGATrWrapper(nn.Module):
    """
    L-GATr for toptagging
    """

    def __init__(
        self,
        net,
        mean_aggregation=False,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.mean_aggregation = mean_aggregation
        self.force_xformers = force_xformers

    def forward(self, batch):
        multivector, scalars = self.embed_into_ga(batch)
        mask = xformers_sa_mask(batch, materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        # embedding happens in the dataset for convenience
        multivector, scalars = batch.x, batch.scalars
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        outputs = extract_scalar(multivector)
        if self.mean_aggregation:
            outputs = outputs.squeeze()
            batchsize = max(batch.batch) + 1
            logits = torch.zeros(batchsize, device=outputs.device, dtype=outputs.dtype)
            logits.index_add_(0, batch.batch, outputs)  # sum
            logits = logits / torch.bincount(batch.batch)  # mean
        else:
            logits = outputs[batch.is_global][:, 0]
        return logits


class TopTaggingCLSGATrWrapper(nn.Module):
    """
    L-CLSGATr for toptagging
    """

    def __init__(
        self,
        net,
        num_class_tokens,
        force_xformers=True,
    ):
        super().__init__()
        self.net = net
        self.num_class_tokens = num_class_tokens
        self.force_xformers = force_xformers
        assert self.num_class_tokens == 1, "num_class_tokens>1 not properly tested yet"

    def forward(self, batch):
        multivector, scalars = self.embed_into_ga(batch)
        batchsize = torch.max(batch.batch).tolist() + 1
        sa_mask = xformers_sa_mask(batch, materialize=not self.force_xformers)
        ca_mask = xformers_ca_mask(
            batch,
            num_class_tokens=self.net.num_class_tokens,
            materialize=not self.force_xformers,
        )
        cls_multivector, cls_scalar = self.net(
            multivector,
            scalars=scalars,
            selfattn_mask=sa_mask,
            crossattn_mask=ca_mask,
            batch=batch.batch,
        )
        logits = self.extract_from_ga(batch, cls_multivector, cls_scalar)

        return logits

    def embed_into_ga(self, batch):
        # embedding happens in the dataset for convenience
        multivector, scalars = batch.x, batch.scalars
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        logits = extract_scalar(multivector)[:, 0, 0]
        return logits
