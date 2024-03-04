import torch
import numpy as np
from torch_geometric.data import Data

from experiments.logger import LOGGER


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
    """

    def __init__(
        self,
        filename,
        mode,
        pairs,
        keep_on_gpu,
        add_jet_momentum=False,
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
        pairs : dataclass
            Dataclass object containing options for adding pairwise tokens
            The pairwise tokens are constructed at runtime to save memory
        keep_on_gpu: bool
            Whether to keep all data on gpu throughout training
        data_scale : float
            std() of all entries in the train dataset
            Effectively a change of units to make the network entries O(1)
        dtype: str
            Not supported consistently
        device: torch.device
            Device on which the dataset will be stored
        """
        self.pairs = pairs
        self.dtype = dtype
        self.device = device
        self.add_jet_momentum = add_jet_momentum

        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        # preprocessing
        if mode == "train":
            data_scale = kinematics.std()
        else:
            assert data_scale is not None
        self.data_scale = data_scale

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
            if self.add_jet_momentum:
                global_token = x.sum(dim=0, keepdim=True)
            else:
                global_token = torch.zeros_like(x[[0], ...], dtype=self.dtype)
                global_token[..., 0] = 1
            x = torch.cat((global_token, x), dim=0)
            is_global = torch.zeros(x.shape[0], 1, dtype=torch.bool)
            is_global[0, 0] = True

            data = Data(x=x, label=label, is_global=is_global)
            if keep_on_gpu:
                data = data.to(self.device)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.pairs.use:
            batch = self.data_list[idx].to(self.device)
            n = (
                batch.x.shape[0] - 1
            )  # number of constituents in the event (first constituent is global token)

            # create single tokens as (fourmomentum, 0, 0)
            num_paired_channels = 3 if self.pairs.delta else 2
            single = torch.concatenate(
                (
                    batch.x.reshape(n + 1, 1, 4),
                    torch.zeros(
                        n + 1,
                        num_paired_channels,
                        4,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                ),
                dim=1,
            )

            # create pairwise tokens as (0, fourmomentum1, fourmomentum2)
            pairs = torch.stack(
                (
                    batch.x[1:, :].reshape(n, 1, 1, 4).expand(n, n, 1, 4),
                    batch.x[1:, :].reshape(1, n, 1, 4).expand(n, n, 1, 4),
                ),
                dim=2,
            )
            if self.pairs.delta:
                # add fourmomentum1 - fourmomentum2 as extra channel
                delta = batch.x[1:, :].reshape(n, 1, 1, 4).expand(n, n, 1, 4) - batch.x[
                    1:, :
                ].reshape(1, n, 1, 4).expand(n, n, 1, 4)
                pairs = torch.cat((pairs, delta.reshape(n, n, 1, 1, 4)), dim=2)
            pairs = torch.concatenate(
                (
                    torch.zeros(n**2, 1, 4, dtype=self.dtype, device=self.device),
                    pairs.reshape(n**2, num_paired_channels, 4),
                ),
                dim=1,
            )
            if self.pairs.directed:
                # remove pairs with fourmomentum1 <= fourmomentum2
                # mask = torch.tensor([idx1 < idx2 for idx1 in range(n) for idx2 in range(n)], device=self.device) # this is slow because of the loop
                idx = torch.triu_indices(n, n, device=self.device)
                mask = torch.ones(n, n, dtype=torch.bool, device=self.device)
                mask[idx[0], idx[1]] = False
                mask = mask.flatten()

                pairs = pairs[mask, ...]
            if self.pairs.top_k is not None:
                # keep only the top_k pairs with highest kt-distance
                p1, p2 = pairs[..., 1, :], pairs[..., 2, :]
                kt = get_kt(p1, p2)
                sort_idx = torch.sort(kt, descending=True)[1]
                pairs = pairs[sort_idx, ...]
                if pairs.shape[0] >= self.pairs.top_k:
                    pairs = pairs[: self.pairs.top_k, ...]

            # combine single and pairwise tokens
            # (and update the is_global mask)
            x = torch.concatenate((single, pairs), dim=0)
            is_global = torch.concatenate(
                (
                    batch.is_global,
                    torch.zeros(
                        pairs.shape[0], 1, dtype=torch.bool, device=self.device
                    ),
                ),
                dim=0,
            )

            # return new object instead of overwriting the old one!
            return Data(x=x, label=batch.label, is_global=is_global)
        else:
            return self.data_list[idx].to(self.device)


def _get_pt(p):
    # transverse momentum
    return torch.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2)


def _get_phi(p):
    # azimuthal angle
    return torch.arctan2(p[..., 2], p[..., 1])


def _get_eta(p):
    # rapidity
    p_abs = torch.sqrt(torch.sum(p[..., 1:] ** 2, dim=-1))
    return torch.arctanh(p[..., 3] / p_abs)


def _get_deltaR(p1, p2):
    # deltaR = angular distance
    phi1, phi2 = _get_phi(p1), _get_phi(p2)
    eta1, eta2 = _get_eta(p1), _get_eta(p2)
    return torch.sqrt((phi1 - phi2) ** 2 + (eta1 - eta2) ** 2)


def get_kt(p1, p2):
    # un-normalized kt distance, corresponding to R=1
    pt1, pt2 = _get_pt(p1), _get_pt(p2)
    deltaR = _get_deltaR(p1, p2)
    return deltaR * torch.min(pt1, pt2)
