import numpy as np
import torch
from torch import nn

from gatr.interface import extract_scalar, embed_vector
from experiments.toptagging.dataset import embed_beam_reference
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

def attention_mask_pretrain(attention_mask_list, force_xformers=True):
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
    bincounts = torch.bincount(attention_mask_list).tolist()
    mask = BlockDiagonalMask.from_seqlens(bincounts)
    if not force_xformers:
        # materialize mask to torch.tensor (only for testing purposes)
        mask = mask.materialize(shape=(len(attention_mask_list), len(attention_mask_list)))
    return mask


class TopTaggingPretrainGATrWrapper(nn.Module):
    """
    GATr for toptagging
    including all kinds of options to play with
    """

    def __init__(
        self,
        net,
        mean_aggregation=False,
        force_xformers=True,
        add_jet_momentum=False,
        beam_reference="spacelike",
        add_time_reference=False,
        two_beams=False,
        beam_token=True,
    ):
        super().__init__()
        self.net = net
        self.mean_aggregation = mean_aggregation
        self.force_xformers = force_xformers
        self.add_jet_momentum = add_jet_momentum
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference
        self.two_beams = two_beams
        self.beam_token = beam_token

    def forward(self, batch):
        # Put the batch into the (batch_size, num_particles, (E, px, py, pz)) format
        batch = torch.transpose(batch, 1, 2)

        # Define the attention mask
        multivector_list = []
        scalar_list = []
        is_global_list = []
        attention_mask_list = []
        for i in range(batch.shape[0]):

            nonzero_indices = (batch[i, ...].abs() > 1e-5).all(dim=-1)
            x = batch[i, ...][nonzero_indices]

            # Add global token
            global_token = torch.zeros_like(x[[0], ...], dtype=batch.dtype, device=batch.device)
            global_token[..., 0] = 1
            x = torch.cat((global_token, x), dim=0)

            # Define is_global index
            is_global = torch.zeros((x.shape[0], 10, 1), dtype=torch.bool, device=batch.device)
            is_global[0] = True

            # create embeddings
            x = x.unsqueeze(1)

            scalars = torch.zeros(
                x.shape[0], 0, device=batch.device, dtype=batch.dtype
            )
            x = embed_vector(x)

            # beam reference
            beam = embed_beam_reference(
                x,
                self.beam_reference,
                self.add_time_reference,
                self.two_beams,
            )

            if beam is not None:
                if self.beam_token:
                    # embed beam as extra token
                    beam = beam.unsqueeze(1)
                    x = torch.cat((x, beam), dim=-3)
                    scalars = torch.cat(
                        (
                            scalars,
                            torch.zeros(
                                beam.shape[0],
                                scalars.shape[1],
                                device=beam.device,
                                dtype=beam.dtype,
                            ),
                        ),
                        dim=-2,
                    )
                    is_global = torch.cat(
                        (
                            is_global,
                            torch.zeros(
                                beam.shape[0],
                                10,
                                1,
                                device=beam.device,
                                dtype=torch.bool,
                            ),
                        ),
                        dim=-3,
                    )
                else:
                    # embed beam as channel for each particle
                    beam = beam.unsqueeze(0).expand(x.shape[0], *beam.shape)
                    x = torch.cat((x, beam), dim=-2)

            # add information about which token is global
            scalars_is_global = torch.zeros(
                scalars.shape[0], 1, device=batch.device, dtype=batch.dtype
            )
            scalars_is_global[0, :] = 1.0
            scalars = torch.cat([scalars_is_global, scalars], dim=-1)

            multivector_list.append(x)
            scalar_list.append(scalars)
            is_global_list.append(is_global)
            attention_mask_list.append(torch.ones_like(x[..., 0], dtype=batch.dtype, device=batch.device)*i)

        multivector = torch.cat(multivector_list).unsqueeze(0)
        scalars = torch.cat(scalar_list).unsqueeze(0)
        is_global = torch.cat(is_global_list)
        attention_mask_list = torch.cat(attention_mask_list)

        # Get the attention mask into the xformers format
        mask = attention_mask_pretrain(attention_mask_list.flatten().int(), self.force_xformers)

        multivector_outputs, scalar_outputs = self.net(
            multivector, scalars=scalars, attention_mask=mask
        )

        logits = self.extract_from_ga(multivector_outputs, scalar_outputs, attention_mask_list, is_global)

        return logits

    def extract_from_ga(self, multivector, scalars, attention_mask_list, is_global):
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
        mask = xformers_sa_mask(batch, materialize=not self.force_xformers)
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
