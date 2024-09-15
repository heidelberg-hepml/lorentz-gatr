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
