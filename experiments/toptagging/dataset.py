import torch

class TopTaggingDataset(torch.utils.data.Dataset):

    def __init__(self, kinematics, labels, dtype):
        self.kinematics = torch.tensor(kinematics, dtype=dtype)
        self.labels = torch.tensor(labels, dtype=dtype)

        # construct attention_mask
        network_content = torch.cat((torch.randn_like(self.kinematics[:,[0],:]), self.kinematics), dim=1) # take global_token into account
        mask_1d = (network_content != 0)[...,0]
        self.mask = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(2)
        self.mask = self.mask.unsqueeze(1) # take multi-headed self-attention into account

        # for some reason pytorch does not like torch.bool mask, so I manually make it a torch.float mask
        # (this is done anyway according to the documentation, but adding this here fixed a nan error for me)
        self.mask.masked_fill_(self.mask.logical_not(), float("-inf")) 

    def __len__(self):
        return self.kinematics.shape[0]

    def __getitem__(self, idx):
        return self.kinematics[idx], self.labels[idx], self.mask[idx]
