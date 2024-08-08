import numpy as np
import torch
from torch import nn

from gatr.interface import extract_scalar
from xformers.ops.fmha import BlockDiagonalMask


def attention_mask(batch, device, force_xformers=True):
    """
    Construct attention mask that makes surfe that objects only attend to each other
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
        mask = mask.materialize(shape=(len(batch.batch), len(batch.batch))).to(device)
    return mask


class TopTaggingGATrWrapper(nn.Module):
    """
    L-GATr for toptagging
    including all kinds of options to play with
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
        mask = attention_mask(batch, scalars.device, self.force_xformers)
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
