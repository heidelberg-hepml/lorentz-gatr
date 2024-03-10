# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import torch
from torch import nn

from torchdiffeq import odeint


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.weights = nn.Parameter(
            scale * torch.randn(input_dim, embed_dim // 2), requires_grad=False
        )

    def forward(self, t):
        projection = 2 * math.pi * torch.matmul(t, self.weights)
        embedding = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        return embedding


class CFM(nn.Module):
    def __init__(
        self,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_t_dim, scale=embed_t_scale),
            nn.Linear(embed_t_dim, embed_t_dim),
        )

    def batch_loss(self, x0, ijet, loss_fn):
        t = torch.rand(x0.shape[0], 1, 1, dtype=x0.dtype, device=x0.device)
        eps = self.sample_base(x0.shape).to(device=x0.device, dtype=x0.dtype)
        x_t = (1 - t) * x0 + t * eps
        v_t = -x0 + eps

        v_pred = self.get_velocity(x_t, t, ijet=ijet)
        loss = loss_fn(v_pred, v_t)
        '''
        lossi = (v_pred - v_t) ** 2
        print(lossi.max(dim=0)[0])
        import matplotlib.pyplot as plt
        for i in range(4):
            plt.hist(v_pred[...,i].flatten().detach(), bins=100, alpha=.5)
            plt.hist(v_t[...,i].flatten().detach(), bins=100, alpha=.5)
            plt.show()
        '''
        return loss

    def sample(self, ijet, shape, device, dtype):
        def velocity(t, x_t):
            t = t * torch.ones(shape[0], 1, 1, dtype=dtype, device=device)
            v_t = self.get_velocity(x_t, t, ijet=ijet)
            return v_t

        eps = self.sample_base(shape).to(device=device, dtype=dtype)
        x_t = odeint(
            velocity,
            eps,
            torch.tensor([1.0, 0.0]),
            # method="rk4",
            # options={"step_size": 1e-2},
        )[-1]
        return x_t

    def log_prob(self, x, ijet):
        raise NotImplementedError

    def get_velocity(self, x, t, ijet):
        raise NotImplementedError
