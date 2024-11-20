import torch
from torch.nn.functional import one_hot
from torch_geometric.utils import scatter

from experiments.tagging.dataset import EPS
from gatr.interface import embed_vector, embed_spurions

UNITS = 20  # We use units of 20 GeV for all tagging experiments


def get_batch_from_ptr(ptr):
    return torch.arange(len(ptr) - 1, device=ptr.device).repeat_interleave(
        ptr[1:] - ptr[:-1],
    )


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
    if cfg_data.add_scalar_features:
        log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
        log_energy = fourmomenta[..., 0].unsqueeze(-1).log()

        batch = get_batch_from_ptr(ptr)
        jet = scatter(fourmomenta, index=batch, dim=0, reduce="sum").index_select(
            0, batch
        )
        log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
        log_energy_rel = (fourmomenta[..., 0].log() - jet[..., 0].log()).unsqueeze(-1)
        phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
        dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
        eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
        deta = -(eta_4 - eta_jet).unsqueeze(-1)
        dr = torch.sqrt(dphi**2 + deta**2)
        scalar_features = [
            log_pt,
            log_energy,
            log_pt_rel,
            log_energy_rel,
            dphi,
            deta,
            dr,
        ]
        for i, feature in enumerate(scalar_features):
            mean, factor = cfg_data.scalar_features_preprocessing[i]
            scalar_features[i] = (feature - mean) * factor
        scalars = torch.cat(
            (scalars, *scalar_features),
            dim=-1,
        )

    # embed fourmomenta into multivectors
    if cfg_data.rescale_data:
        fourmomenta /= UNITS
    multivectors = embed_vector(fourmomenta)
    multivectors = multivectors.unsqueeze(-2)

    # beam reference
    spurions = embed_spurions(
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
        spurion_idxs = torch.stack(
            [ptr[:-1] + i for i in range(n_spurions)], dim=0
        ) + n_spurions * torch.arange(batchsize, device=ptr.device)
        spurion_idxs = spurion_idxs.permute(1, 0).flatten()
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


def get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def get_phi(p):
    # azimuthal angle
    return torch.arctan2(p[..., 2], p[..., 1])


def get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    return torch.arctanh(p[..., 3] / p_abs)
