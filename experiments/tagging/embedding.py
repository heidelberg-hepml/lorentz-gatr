import torch
from torch.nn.functional import one_hot

from experiments.tagging.dataset import EPS
from gatr.interface import embed_vector


def embed_tagging_data_into_ga(fourmomenta, scalars, ptr, cfg_data):
    """
    Embed tagging data into geometric algebra representation
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    scalars: torch.tensor of shape (n_particles, n_features)
        Optional scalar features, n_features=0 is possible
    ptr: torch.tensor of shape (batchsize+1)
        Indices of the first particle for each jet
        Also includes the first index after the batch ends
    cfg_data: settings for embedding

    Returns
    -------
    embedding: dict
        Embedded data
        Includes keys for multivectors, scalars, is_global and ptr
    """
    batchsize = len(ptr) - 1
    arange = torch.arange(batchsize, device=fourmomenta.device)

    # add extra scalar channels
    if cfg_data.add_pt:
        log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
        scalars = torch.cat((scalars, log_pt), dim=-1)
    if cfg_data.add_energy:
        log_energy = fourmomenta[..., 0].unsqueeze(-1).log()
        scalars = torch.cat((scalars, log_energy), dim=-1)

    # embed fourmomenta into multivectors
    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    # beam reference
    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        cfg_data.add_xzplane,
        cfg_data.add_yzplane,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    n_spurions = spurions.shape[0]
    if cfg_data.beam_token:
        # prepend spurions to the token list (within each block)
        spurion_idxs = torch.cat(
            [
                torch.arange(
                    ptr_start + i * n_spurions,
                    ptr_start + (i + 1) * n_spurions,
                )
                for i, ptr_start in enumerate(ptr[:-1])
            ]
        )
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
            [
                torch.arange(
                    ptr_start + i * num_global_tokens,
                    ptr_start + (i + 1) * num_global_tokens,
                )
                for i, ptr_start in enumerate(ptr[:-1])
            ]
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


def dense_to_sparse_jet(fourmomenta_dense, scalars_dense):
    """
    Transform dense jet into sparse jet

    Parameters
    ----------
    fourmomenta_dense: torch.tensor of shape (batchsize, 4, num_particles_max)
    scalars_dense: torch.tensor of shape (batchsize, num_features, num_particles_max)

    Returns
    -------
    fourmomenta_sparse: torch.tensor of shape (num_particles, 4)
        Fourmomenta for concatenated list of particles of all jets
    scalars_sparse: torch.tensor of shape (num_particles, num_features)
        Scalar features for concatenated list of particles of all jets
    ptr: torch.tensor of shape (batchsize+1)
        Start indices of each jet, this way we don't lose information when concatenating everything
        Starts with 0 and ends with the first non-accessible index (=total number of particles)
    """
    fourmomenta_dense = torch.transpose(
        fourmomenta_dense, 1, 2
    )  # (batchsize, num_particles, 4)
    scalars_dense = torch.transpose(
        scalars_dense, 1, 2
    )  # (batchsize, num_particles, num_features)

    mask = (fourmomenta_dense.abs() > EPS).any(dim=-1)
    num_particles = mask.sum(dim=-1)
    fourmomenta_sparse = fourmomenta_dense[mask]
    scalars_sparse = scalars_dense[mask]

    ptr = torch.zeros(
        len(num_particles) + 1, device=fourmomenta_dense.device, dtype=torch.long
    )
    ptr[1:] = torch.cumsum(num_particles, dim=0)
    return fourmomenta_sparse, scalars_sparse, ptr


def get_spurion(
    beam_reference,
    add_time_reference,
    two_beams,
    add_xzplane,
    add_yzplane,
    device,
    dtype,
):
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
    add_xzplane: bool
        Whether to add the x-z-plane as a reference to the network
    add_yzplane: bool
        Whether to add the y-z-plane as a reference to the network
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
        beam = torch.empty(0, 16, device=device, dtype=dtype)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_xzplane:
        # add the x-z-plane, embedded as a bivector
        xzplane = torch.zeros(1, 16, device=device, dtype=dtype)
        xzplane[..., 10] = 1
    else:
        xzplane = torch.empty(0, 16, device=device, dtype=dtype)

    if add_yzplane:
        # add the y-z-plane, embedded as a bivector
        yzplane = torch.zeros(1, 16, device=device, dtype=dtype)
        yzplane[..., 9] = 1
    else:
        yzplane = torch.empty(0, 16, device=device, dtype=dtype)

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=device, dtype=dtype).reshape(1, 4)
        time = embed_vector(time)
    else:
        time = torch.empty(0, 16, device=device, dtype=dtype)

    spurion = torch.cat((beam, xzplane, yzplane, time), dim=-2)
    return spurion


def get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)
