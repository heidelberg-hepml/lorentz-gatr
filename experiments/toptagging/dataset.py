import torch
import numpy as np
from torch_geometric.data import Data

from gatr.interface import embed_vector


class TopTaggingDataset(torch.utils.data.Dataset):

    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    label : torch.tensor of shape (1), dtype torch.int
        label of the jet (0=QCD, 1=top)
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    def __init__(
        self,
        filename,
        mode,
        cfg,
        data_scale=None,
        dtype=torch.float,
        device="cpu",
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        cfg : dataclass
            Dataclass object containing options for the format
            in which the data is preprocessed at runtime
        data_scale : float
            std() of all entries in the train dataset
            Effectively a change of units to make the network entries O(1)
        dtype: str
            Not supported consistently
        device: torch.device
            Device on which the dataset will be stored
        """
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        # preprocessing
        if mode == "train":
            data_scale = kinematics.std()
        else:
            assert data_scale is not None
        self.data_scale = data_scale

        if self.cfg.data.rescale_data:
            kinematics = kinematics / self.data_scale
        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > 1e-5).all(dim=-1)
            x = kinematics[i, ...][mask]
            label = labels[i, ...]

            # construct global token
            if self.cfg.data.include_global_token:
                global_token = torch.zeros_like(x[[0], ...], dtype=self.dtype)
                global_token[..., 0] = 1
                x = torch.cat((global_token, x), dim=0)
                is_global = torch.zeros(x.shape[0], 1, dtype=torch.bool)
                is_global[0, 0] = True
            else:
                is_global = None
            data = Data(x=x, label=label, is_global=is_global)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        batch = self.data_list[idx].to(self.device)

        # create embeddings
        x = batch.x.unsqueeze(1)
        scalars = torch.zeros(x.shape[0], 0, device=self.device, dtype=self.dtype)
        if self.cfg.data.add_pt:
            pt = get_pt(x)
            scalars = torch.cat((scalars, pt), dim=-1)
        is_global = batch.is_global if self.cfg.data.include_global_token else None

        # beam reference
        x = embed_vector(x)
        beam = embed_beam_reference(
            x,
            self.cfg.data.beam_reference,
            self.cfg.data.add_time_reference,
            self.cfg.data.two_beams,
        )
        if beam is not None:
            if self.cfg.data.beam_token:
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
                if self.cfg.data.include_global_token:
                    is_global = torch.cat(
                        (
                            is_global,
                            torch.zeros(
                                beam.shape[0],
                                1,
                                device=beam.device,
                                dtype=torch.bool,
                            ),
                        ),
                        dim=-2,
                    )

            else:
                # embed beam as channel for each particle
                beam = beam.unsqueeze(0).expand(x.shape[0], *beam.shape)
                x = torch.cat((x, beam), dim=-2)

        # add information about which token is global
        if self.cfg.data.include_global_token:
            scalars_is_global = torch.zeros(
                scalars.shape[0], 1, device=self.device, dtype=self.dtype
            )
            scalars_is_global[0, :] = 1.0
            scalars = torch.cat([scalars_is_global, scalars], dim=-1)

        return Data(x=x, scalars=scalars, label=batch.label, is_global=is_global)


class QGTaggingDataset(torch.utils.data.Dataset):

    """
    We use torch_geometric to handle point cloud of jet constituents more efficiently
    The torch_geometric dataloader concatenates jets along their constituent direction,
    effectively combining the constituent index with the batch index in a single dimension.
    An extra object batch.batch for each batch specifies to which jet the constituent belongs.
    We extend the constituent list by a global token that is used to embed extra global
    information and extract the classifier score.

    Structure of the elements in self.data_list
    x : torch.tensor of shape (num_elements, 4)
        List of 4-momenta of jet constituents
    pid : torch.tensor of shape (num_elements, 6), dtype torch.int
        List of particle identification variables of jet constituents
    label : torch.tensor of shape (1), dtype torch.int
        label of the jet (0=quark, 1=gluon)
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    def __init__(
        self,
        filename,
        mode,
        cfg,
        data_scale=None,
        dtype=torch.float,
        device="cpu",
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        cfg : dataclass
            Dataclass object containing options for the format
            in which the data is preprocessed at runtime
        data_scale : float
            std() of all entries in the train dataset
            Effectively a change of units to make the network entries O(1)
        dtype: str
            Not supported consistently
        device: torch.device
            Device on which the dataset will be stored
        """
        self.cfg = cfg
        self.dtype = dtype
        self.device = device

        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        pids = data[f"pid_{mode}"]
        labels = data[f"labels_{mode}"]

        # preprocessing
        if mode == "train":
            data_scale = kinematics.std()
        else:
            assert data_scale is not None
        self.data_scale = data_scale

        if self.cfg.data.rescale_data:
            kinematics = kinematics / self.data_scale
        kinematics = torch.tensor(kinematics, dtype=dtype)
        pids = torch.tensor(pids, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > 1e-5).all(dim=-1)
            x = kinematics[i, ...][mask]
            pid = pids[i, ...][mask]
            label = labels[i, ...]

            # construct global token
            if self.cfg.data.include_global_token:
                global_token = torch.zeros_like(x[[0], ...], dtype=self.dtype)
                global_token[..., 0] = 1
                x = torch.cat((global_token, x), dim=0)
                is_global = torch.zeros(x.shape[0], 1, dtype=torch.bool)
                is_global[0, 0] = True
            else:
                is_global = None

            global_token_pid = torch.zeros_like(pid[[0], ...], dtype=self.dtype)
            pid = torch.cat((global_token_pid, pid), dim=0)

            data = Data(x=x, pid=pid, label=label, is_global=is_global)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        batch = self.data_list[idx].to(self.device)

        # create embeddings
        x = batch.x.unsqueeze(1)
        scalars = batch.pid
        if self.cfg.data.add_pt:
            pt = get_pt(x)
            scalars = torch.cat((scalars, pt), dim=-1)
        is_global = batch.is_global if self.cfg.data.include_global_token else None

        # beam reference
        x = embed_vector(x)
        beam = embed_beam_reference(
            x,
            self.cfg.data.beam_reference,
            self.cfg.data.add_time_reference,
            self.cfg.data.two_beams,
        )
        if beam is not None:
            if self.cfg.data.beam_token:
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
                if self.cfg.data.include_global_token:
                    is_global = torch.cat(
                        (
                            is_global,
                            torch.zeros(
                                beam.shape[0],
                                1,
                                device=beam.device,
                                dtype=torch.bool,
                            ),
                        ),
                        dim=-2,
                    )

            else:
                # embed beam as channel for each particle
                beam = beam.unsqueeze(0).expand(x.shape[0], *beam.shape)
                x = torch.cat((x, beam), dim=-2)

        # add information about which token is global
        if self.cfg.data.include_global_token:
            scalars_is_global = torch.zeros(
                scalars.shape[0], 1, device=self.device, dtype=self.dtype
            )
            scalars_is_global[0, :] = 1.0
            scalars = torch.cat([scalars_is_global, scalars], dim=-1)
        return Data(x=x, scalars=scalars, label=batch.label, is_global=is_global)


def embed_beam_reference(p_ref, beam_reference, add_time_reference, two_beams):
    """
    Construct symmetry-breaking reference multivector

    Parameters
    ----------
    p_ref: torch.tensor with shape (..., items, 4)
        Reference tensor to infer device, dtype and shape for the beam_reference
    beam_reference: str
        Different options for adding a beam_reference
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam

    Returns
    -------
    beam: torch.tensor with shape (..., items, mv_channels, 16)
        beam embedded as mv_channels multivectors
    """

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, device=p_ref.device, dtype=p_ref.dtype).reshape(1, 4)
        beam = embed_vector(beam)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 4] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)

    elif beam_reference == "xyplane":
        # add the x-y-plane, embedded as a bivector
        # convention for bivector components: [tx, ty, tz, xy, xz, yz]
        beam = torch.zeros(1, 16, device=p_ref.device, dtype=p_ref.dtype)
        beam[..., 8] = 1

    elif beam_reference is None:
        beam = None

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=p_ref.device, dtype=p_ref.dtype).reshape(1, 4)
        time = embed_vector(time)
        if beam is None:
            reference = time
        else:
            reference = torch.cat([beam, time], dim=-2)
    else:
        reference = beam

    return reference


def get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def _get_phi(p):
    # azimuthal angle
    return torch.arctan2(p[..., 2], p[..., 1])


def _get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    return torch.arctanh(p[..., 3] / p_abs)


def get_deltaR(p1, p2):
    # deltaR = angular distance
    phi1, phi2 = _get_phi(p1), _get_phi(p2)
    eta1, eta2 = _get_eta(p1), _get_eta(p2)
    return torch.sqrt((phi1 - phi2) ** 2 + (eta1 - eta2) ** 2)


def get_kt(p1, p2):
    # un-normalized kt distance, corresponding to R=1
    pt1, pt2 = get_pt(p1), get_pt(p2)
    deltaR = get_deltaR(p1, p2)
    return deltaR * torch.min(pt1, pt2)
