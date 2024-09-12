import numpy as np
import torch
from torch import nn

from gatr.interface import extract_scalar, embed_vector
from xformers.ops.fmha import BlockDiagonalMask

def xformers_sa_mask(batch_indices, materialize=False):
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
    bincounts = torch.bincount(batch_indices).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if materialize:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(batch_indices), len(batch_indices))).to(
            batch_indices.device
        )
    return mask


class TopTaggingPretrainGATrWrapper(nn.Module):
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

    def forward(self, multivector, scalars, is_global, attention_indices):
        """
        # Put the batch into the (batch_size, num_particles, (E, px, py, pz)) format
        EPS = 1e-5

        batch = torch.transpose(batch, 1, 2)

        mask_num_particles = (batch.abs() > EPS).any(dim=-1)
        num_particles = mask_num_particles.sum(dim=-1)

        # flatten the events dimension and remove zero padding
        nonzero_mask = (batch.abs()>EPS).any(dim=-1)
        batch_sparse = batch[nonzero_mask]

        # create batch.ptr from torch_geometric.data.Data by hand
        ptr = torch.zeros_like(num_particles, device=batch.device)
        ptr[1:] = torch.cumsum(num_particles, dim=0)[:-1]

        # insert global token at beginning of batch
        batchsize = len(ptr)
        ptr_with_global = ptr + torch.arange(batchsize, device=batch.device)
        is_global = torch.zeros(batch_sparse.shape[0] + batchsize, *batch_sparse.shape[1:], dtype=torch.bool, device=batch.device)
        is_global[ptr_with_global] = True
        batch_sparse_with_global = torch.zeros_like(is_global, dtype=batch_sparse.dtype)
        batch_sparse_with_global[~is_global] = batch_sparse.flatten()
        is_global = is_global[:, [0]]

        # define the attention indices for the events in the batch as in batch.batch from torch_geometric.data.Data
        get_batch_from_ptr = lambda ptr: torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(ptr[1:] - ptr[:-1])
        attention_mask_list = get_batch_from_ptr(ptr_with_global)

        batch_sparse_with_global = embed_vector(batch_sparse_with_global)

        beam = embed_beam_reference(
            batch,
            self.beam_reference,
            self.add_time_reference,
            self.two_beams,
        )

        batch_final = batch_sparse_with_global
        is_global_final = is_global

        if beam is not None:
            if self.beam_token:
                # embed beam as extra token
                ptr_beam = torch.cumsum(num_particles, dim=0) - 1
                ptr_beam_total = torch.cat([ptr_beam + (beam.shape[0]+1)*torch.arange(1, batchsize + 1, device=batch.device) - i for i in range(beam.shape[0])])

                batch_final = torch.zeros((batch_sparse_with_global.shape[0] + beam.shape[0] * batchsize, *batch_sparse_with_global.shape[1:]), device=batch.device)
                mask_beam = torch.zeros_like(batch_final[..., 0], dtype=torch.bool, device=batch.device)
                mask_beam[ptr_beam_total] = True

                batch_final[mask_beam] = beam.repeat(batchsize, 1)
                batch_final[~mask_beam] = batch_sparse_with_global
                batch_final = batch_final.unsqueeze(-2)

                #Extend is_global and the attention_mask_list to include the extra token
                is_global_final = torch.zeros((batch_sparse_with_global.shape[0] + beam.shape[0] * batchsize, 1), dtype=torch.bool, device=batch.device)
                is_global_final[~mask_beam] = is_global

                attention_mask_list = get_batch_from_ptr(ptr_with_global + beam.shape[0] * torch.arange(batchsize, device=batch.device))
            else:
                batch_sparse_with_global = batch_sparse_with_global.unsqueeze(-2)
                beam = beam.unsqueeze(0).repeat(batch_sparse_with_global.shape[0], 1, 1)
                batch_final = torch.cat((batch_sparse_with_global, beam), dim=-2)
                is_global_final=is_global

        multivector = batch_final.unsqueeze(0)
        scalars = is_global_final.float().unsqueeze(0)
        is_global_final = is_global_final.unsqueeze(-2).expand(batch_final.shape[0], 10, 1)
        attention_mask_list = torch.cat((attention_mask_list, (batchsize - 1) * torch.ones(len(batch_final) - len(attention_mask_list), device=batch.device)), dim=0)
        """

        # Get the attention mask into the xformers format
        mask = xformers_sa_mask(attention_indices.int(), materialize=not self.force_xformers)

        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )

        logits = self.extract_from_ga(multivector_outputs, scalar_outputs, is_global)

        return logits

    def extract_from_ga(self, multivector, scalars, is_global):
        outputs = extract_scalar(multivector).squeeze()
        labels = outputs.unsqueeze(-1)[is_global]

        return labels


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
        mask = xformers_sa_mask(batch.batch, materialize=not self.force_xformers)
        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )
        logits = self.extract_from_ga(batch, multivector_outputs, scalar_outputs)

        return logits

    def embed_into_ga(self, batch):
        # embedding happens in the dataset for convenience
        # add artificial batch index (otherwise xformers attention on gpu complains)
        multivector, scalars = batch.x.unsqueeze(0), batch.scalars.unsqueeze(0)
        return multivector, scalars

    def extract_from_ga(self, batch, multivector, scalars):
        outputs = extract_scalar(multivector)
        if self.mean_aggregation:
            outputs = outputs[0, :, 0, 0]
            batchsize = max(batch.batch) + 1
            logits = torch.zeros(batchsize, device=outputs.device, dtype=outputs.dtype)
            logits.index_add_(0, batch.batch, outputs)  # sum
            logits = logits / torch.bincount(batch.batch)  # mean
        else:
            logits = outputs[0, :, :, 0][batch.is_global]
        return logits
