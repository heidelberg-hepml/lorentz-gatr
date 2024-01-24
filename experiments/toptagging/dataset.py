import torch
import numpy as np
from torch_geometric.data import Data

class TopTaggingDataset(torch.utils.data.Dataset):

    def __init__(self, filename, mode, data_scale=None, dtype=torch.float):
        data = np.load(filename)
        kinematics = data[f"kinematics_{mode}"]
        labels = data[f"labels_{mode}"]

        # preprocessing
        if data_scale is None:
            data_scale = kinematics.std()
        self.data_scale = data_scale
        kinematics = kinematics / self.data_scale
        kinematics = torch.tensor(kinematics, dtype=dtype)
        labels = torch.tensor(labels, dtype=dtype)

        # create list of torch_geometric.data.Data objects
        self.data_list = []
        for i in range(kinematics.shape[0]):
            # drop zero-padded components
            mask = (kinematics[i,...] != 0).any(dim=-1)
            x = kinematics[i,...][mask]
            label = labels[i,...]

            # construct global token
            global_token = torch.zeros_like(x[[0], ...])
            global_token[...,0] = 1
            x = torch.cat((global_token, x), dim=0)
            is_global = torch.zeros(x.shape[0], 1, dtype=torch.bool)
            is_global[0,0] = True

            data = Data(x=x, label=label, is_global=is_global)
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
