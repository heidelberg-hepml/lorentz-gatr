import torch
from torch.nn.functional import one_hot

from gatr.interface import embed_vector


def embed_tagging_data_into_ga(batch, cfg_data):
    """
    Embed tagging data into sparse geometric algebra representation
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    batch: torch_geometric.data.GlobalStorage
        Object that contains information about tagging data
        fourmomenta (in 'x'), scalars, labels
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    device
    dtype

    Returns
    -------
    spurion: torch.tensor with shape (n_spurions, 16)
        spurion embedded as multivector object
    """
    fourmomenta = batch.x
    scalars = batch.scalars
    ptr = batch.ptr
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    # add pt to scalars
    if cfg_data.add_pt:
        pt = get_pt(fourmomenta).unsqueeze(-1)
        scalars = torch.cat((scalars, pt), dim=-1)

    # embed fourmomenta into multivectors
    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    # beam reference
    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]
    if cfg_data.beam_token:
        # prepend spurions to the token list (within each block)
        spurion_idxs = torch.cat([ptr[:-1] + arange + i for i in range(n_spurions)])
        insert_spurion = torch.zeros(
            multivectors.shape[0] + n_spurions * batchsize,
            dtype=torch.bool,
            device=multivectors.device,
        )
        insert_spurion[spurion_idxs] = True
        multivectors_buffer = multivectors.clone()
        multivectors = torch.empty(
            insert_spurion.shape[0],
            *multivectors.shape[1:],
            dtype=multivectors.dtype,
            device=multivectors.device,
        )
        multivectors[~insert_spurion] = multivectors_buffer
        multivectors[insert_spurion] = spurions.repeat(batchsize, 1).unsqueeze(-2)
        scalars_buffer = scalars.clone()
        scalars = torch.zeros(
            multivectors.shape[0],
            scalars.shape[1],
            dtype=scalars.dtype,
            device=scalars.device,
        )
        scalars[~insert_spurion] = scalars_buffer
        ptr[1:] = ptr[1:] + (arange + 1) * n_spurions
    else:
        # append spurion to multivector channels
        spurions = spurions.unsqueeze(0).repeat(multivectors.shape[0], 1, 1)
        multivectors = torch.cat((multivectors, spurions), dim=-2)

    # global tokens
    if cfg_data.include_global_token:
        # prepend global tokens to the token list
        num_global_tokens = cfg_data.num_global_tokens
        global_idxs = torch.cat(
            [ptr[:-1] + arange + i for i in range(num_global_tokens)]
        )
        is_global = torch.zeros(
            multivectors.shape[0] + batchsize * num_global_tokens,
            dtype=torch.bool,
            device=multivectors.device,
        )
        is_global[global_idxs] = True
        multivectors_buffer = multivectors.clone()
        multivectors = torch.zeros(
            is_global.shape[0],
            *multivectors.shape[1:],
            dtype=multivectors.dtype,
            device=multivectors.device,
        )
        multivectors[~is_global] = multivectors_buffer
        scalars_buffer = scalars.clone()
        scalars = torch.zeros(
            multivectors.shape[0],
            scalars.shape[1] + num_global_tokens,
            dtype=scalars.dtype,
            device=scalars.device,
        )
        token_idx = one_hot(torch.arange(num_global_tokens, device=scalars.device))
        token_idx = token_idx.repeat(batchsize, 1)
        scalars[~is_global] = torch.cat(
            (
                scalars_buffer,
                torch.zeros(
                    scalars_buffer.shape[0],
                    token_idx.shape[1],
                    dtype=scalars.dtype,
                    device=scalars.device,
                ),
            ),
            dim=-1,
        )
        scalars[is_global] = torch.cat(
            (
                torch.zeros(
                    token_idx.shape[0],
                    scalars_buffer.shape[1],
                    dtype=scalars.dtype,
                    device=scalars.device,
                ),
                token_idx,
            ),
            dim=-1,
        )
        ptr[1:] = ptr[1:] + (arange + 1) * num_global_tokens
    else:
        is_global = None

    # return dict
    get_batch_from_ptr = lambda ptr: torch.arange(
        len(ptr) - 1, device=multivectors.device
    ).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )
    batch = get_batch_from_ptr(ptr)
    embedding = {
        "mv": multivectors,
        "s": scalars,
        "is_global": is_global,
        "batch": batch,
    }
    return embedding


def jc_batch_encoding(self, batch):
    # Put the batch into the (batch_size, num_particles, (E, px, py, pz)) format
    batch = torch.transpose(batch, 1, 2)

    # flatten the events dimension and remove zero padding
    EPS = 1e-5
    nonzero_mask = (batch.abs() > EPS).any(dim=-1)
    num_particles = nonzero_mask.sum(dim=-1)
    batch_sparse = batch[nonzero_mask]

    # create batch.ptr from torch_geometric.data.Data by hand
    ptr = torch.zeros_like(num_particles, device=batch.device)
    ptr[1:] = torch.cumsum(num_particles, dim=0)[:-1]

    # insert global token at the beginning of the batch
    batchsize = len(ptr)
    ptr_with_global = ptr + torch.arange(batchsize, device=batch.device)
    is_global = torch.zeros(
        batch_sparse.shape[0] + batchsize,
        *batch_sparse.shape[1:],
        dtype=torch.bool,
        device=batch.device,
    )
    is_global[ptr_with_global] = True
    batch_sparse_with_global = torch.zeros_like(is_global, dtype=batch.dtype)
    batch_sparse_with_global[~is_global] = batch_sparse.flatten()
    is_global = is_global[:, [0]]

    # define the attention indices for the events in the batch as in batch.batch from torch_geometric.data.Data
    get_batch_from_ptr = lambda ptr: torch.arange(
        len(ptr) - 1, device=ptr.device
    ).repeat_interleave(ptr[1:] - ptr[:-1])
    attention_indices = get_batch_from_ptr(ptr_with_global)

    # embed batches into the GA
    batch_sparse_with_global = embed_vector(batch_sparse_with_global)

    # Include beam information to the events
    beam = get_spurion(
        self.cfg.data.beam_reference,
        self.cfg.data.add_time_reference,
        self.cfg.data.two_beams,
        device=batch_sparse_with_global.device,
        dtype=batch_sparse_with_global.dtype,
    )

    batch_final = batch_sparse_with_global
    is_global_final = is_global

    if beam is not None:
        if self.cfg.data.beam_token:
            # embed beam as extra token
            ptr_beam = torch.cumsum(num_particles, dim=0) - 1
            ptr_beam_total = torch.cat(
                [
                    ptr_beam
                    + (beam.shape[0] + 1)
                    * torch.arange(1, batchsize + 1, device=batch.device)
                    - i
                    for i in range(beam.shape[0])
                ]
            )

            batch_final = torch.zeros(
                (
                    batch_sparse_with_global.shape[0] + beam.shape[0] * batchsize,
                    *batch_sparse_with_global.shape[1:],
                ),
                device=batch.device,
            )
            mask_beam = torch.zeros_like(
                batch_final[..., 0], dtype=torch.bool, device=batch.device
            )
            mask_beam[ptr_beam_total] = True

            batch_final[mask_beam] = beam.repeat(batchsize, 1)
            batch_final[~mask_beam] = batch_sparse_with_global
            batch_final = batch_final.unsqueeze(-2)

            # extend is_global and the attention_indices to include the extra token
            is_global_final = torch.zeros(
                (batch_sparse_with_global.shape[0] + beam.shape[0] * batchsize, 1),
                dtype=torch.bool,
                device=batch.device,
            )
            is_global_final[~mask_beam] = is_global

            attention_indices = get_batch_from_ptr(
                ptr_with_global
                + beam.shape[0] * torch.arange(batchsize, device=batch.device)
            )
        else:
            batch_sparse_with_global = batch_sparse_with_global.unsqueeze(-2)
            beam = beam.unsqueeze(0).repeat(batch_sparse_with_global.shape[0], 1, 1)
            batch_final = torch.cat((batch_sparse_with_global, beam), dim=-2)
            is_global_final = is_global

    multivector = batch_final.unsqueeze(0)
    scalars = is_global_final.float().unsqueeze(0)

    # change the shape of is_global to accommodate the number of classes in the output of the model
    is_global_final = is_global_final.unsqueeze(-2).expand(
        batch_final.shape[0], self.cfg.jc_params.num_classes, 1
    )

    # modify the attention_indices to include the indices for the last event in the batch
    attention_indices = torch.cat(
        (
            attention_indices,
            (batchsize - 1)
            * torch.ones(
                len(batch_final) - len(attention_indices), device=batch.device
            ),
        ),
        dim=0,
    )

    return multivector, scalars, is_global_final, attention_indices


def get_spurion(beam_reference, add_time_reference, two_beams, device, dtype):
    """
    Construct spurion

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    device
    dtype

    Returns
    -------
    spurion: torch.tensor with shape (n_spurions, 16)
        spurion embedded as multivector object
    """

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, device=device, dtype=dtype).reshape(1, 4)
        beam = embed_vector(beam)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 4] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)

    elif beam_reference == "xyplane":
        # add the x-y-plane, embedded as a bivector
        # convention for bivector components: [tx, ty, tz, xy, xz, yz]
        beam = torch.zeros(1, 16, device=device, dtype=dtype)
        beam[..., 8] = 1

    elif beam_reference is None:
        beam = torch.empty(0, 16)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=device, dtype=dtype).reshape(1, 4)
        time = embed_vector(time)
    else:
        time = torch.empty(0, 16, device=device, dtype=dtype)

    spurion = torch.cat((beam, time), dim=-2)
    return spurion


def get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
