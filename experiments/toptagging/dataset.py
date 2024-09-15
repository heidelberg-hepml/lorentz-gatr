import torch
import numpy as np
from torch_geometric.data import Data


class TaggingDataset(torch.utils.data.Dataset):
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
    scalars : empty placeholder
    label : torch.tensor of shape (1), dtype torch.int
        label of the jet (0=QCD, 1=top)
    is_global : torch.tensor of shape (num_elements), dtype torch.bool
        True for the global token (first element in constituent list), False otherwise
        We set is_global=None if no global token is used
    """

    def __init__(self, rescale_data):
        super().__init__()
        self.rescale_data = rescale_data

    def load_data(self, filename, mode, data_scale):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TopTaggingDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        data_scale=None,
        dtype=torch.float32,
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
        """
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        # preprocessing
        if mode == "train":
            data_scale = kinematics.std()
        else:
            assert data_scale is not None
        self.data_scale = data_scale

        if self.rescale_data:
            kinematics = kinematics / self.data_scale
        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > 1e-5).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            label = labels[i, ...]
            scalars = torch.zeros(
                fourmomenta.shape[0],
                0,
                dtype=dtype,
            )  # no scalar information
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)


class QGTaggingDataset(TaggingDataset):
    def load_data(
        self,
        filename,
        mode,
        data_scale=None,
        dtype=torch.float32,
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
        """
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

        if self.rescale_data:
            kinematics = kinematics / self.data_scale
        kinematics = torch.tensor(kinematics, dtype=dtype)
        pids = torch.tensor(pids, dtype=dtype)
        labels = torch.tensor(labels, dtype=torch.bool)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i, ...].abs() > 1e-5).all(dim=-1)
            fourmomenta = kinematics[i, ...][mask]
            scalars = pids[i, ...][mask]  # PID scalar information
            label = labels[i, ...]
            data = Data(x=fourmomenta, scalars=scalars, label=label)
            self.data_list.append(data)
