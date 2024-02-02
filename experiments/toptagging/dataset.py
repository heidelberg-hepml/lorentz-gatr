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
        self, filename, mode, data_scale=None, add_pairs=False, dtype=torch.float
    ):
        """
        Parameters
        ----------
        filename : str
            Path to file in npz format where the dataset in stored
        mode : {"train", "test", "val"}
            Purpose of the dataset
            Train, test and validation datasets are already seperated in the specified file
        data_scale : float
            std() of all entries in the train dataset
            Effectively a change of units to make the network entries O(1)
        add_pairs : bool
            Option to extend the data by pairwise tokens
            The pairwise tokens are constructed at runtime to save memory
        """
        self.add_pairs = add_pairs
        self.dtype = dtype

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
            global_token = torch.zeros_like(x[[0], ...], dtype=self.dtype)
            global_token[..., 0] = 1
            x = torch.cat((global_token, x), dim=0)
            is_global = torch.zeros(x.shape[0], 1, dtype=torch.bool)
            is_global[0, 0] = True

            data = Data(x=x, label=label, is_global=is_global)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.add_pairs:
            batch = self.data_list[idx]
            n = (
                batch.x.shape[0] - 1
            )  # number of constituents in the event (first constituent is global token)
            is_global = torch.concatenate(
                (batch.is_global, torch.zeros(n**2, 1, dtype=torch.bool)), dim=0
            )

            # combine single and pairwise tokens in channels in the following way:
            # single tokens: (fourmomentum, 0, 0)
            # pairwise tokens: (0, fourmomentum1, fourmomentum2)
            single = torch.concatenate(
                (
                    batch.x.reshape(n + 1, 1, 4),
                    torch.zeros(n + 1, 2, 4, dtype=self.dtype),
                ),
                dim=1,
            )
            pairs = torch.stack(
                (
                    batch.x[1:, :].reshape(n, 1, 1, 4).expand(n, n, 1, 4),
                    batch.x[1:, :].reshape(1, n, 1, 4).expand(n, n, 1, 4),
                ),
                dim=2,
            ).reshape(n**2, 2, 4)
            pairs = torch.concatenate(
                (torch.zeros(n**2, 1, 4, dtype=self.dtype), pairs), dim=1
            )
            x = torch.concatenate((single, pairs), dim=0)

            # return new object instead of overwriting the old one!
            return Data(x=x, label=batch.label, is_global=is_global)
        else:
            return self.data_list[idx]
