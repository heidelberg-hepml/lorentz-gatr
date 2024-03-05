# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import numpy as np
import torch
from torch import nn

from gatr.interface import embed_vector, extract_vector
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity


def get_type_token(x_ref, type_token_channels):
    type_token_raw = torch.arange(x_ref.shape[1], device=x_ref.device, dtype=torch.long)
    type_token = nn.functional.one_hot(type_token_raw, num_classes=type_token_channels)
    type_token = type_token.unsqueeze(0).expand(
        x_ref.shape[0], x_ref.shape[1], type_token_channels
    )
    return type_token


def get_process_token(x_ref, ijet, process_token_channels):
    process_token_raw = torch.tensor([ijet], device=x_ref.device, dtype=torch.long)
    process_token = nn.functional.one_hot(
        process_token_raw, num_classes=process_token_channels
    ).squeeze()
    process_token = process_token.unsqueeze(0).expand(
        x_ref.shape[1], process_token_channels
    )
    process_token = process_token.unsqueeze(0).expand(
        x_ref.shape[0], x_ref.shape[1], process_token_channels
    )
    return process_token


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.weights = nn.Parameter(
            scale * torch.randn(input_dim, embed_dim // 2), requires_grad=False
        )

    def forward(self, t):
        projection = 2 * np.pi * torch.matmul(t, self.weights)
        embedding = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        return embedding


class ttbarTransformerWrapper(nn.Module):
    def __init__(
        self,
        net,
        is_onshell,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
    ):
        super().__init__()
        self.net = net
        self.is_onshell = is_onshell
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_t_dim, scale=embed_t_scale),
            nn.Linear(embed_t_dim, embed_t_dim),
        )

    def sample_base(self, shape, gen=None):
        eps = torch.randn(shape, generator=gen)
        eps[..., 0] = torch.abs(eps[..., 0])
        return eps

    def forward(self, x, t, ijet):
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x = torch.cat([x, type_token, process_token, t_embedding], dim=2)

        v = self.net(x)
        v[..., self.is_onshell, 3] = 0.0
        return v


class ttbarGATrWrapper(nn.Module):
    def __init__(
        self,
        net,
        is_onshell,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
    ):
        super().__init__()
        self.net = net
        self.is_onshell = is_onshell
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_t_dim, scale=embed_t_scale),
            nn.Linear(embed_t_dim, embed_t_dim),
        )

    def sample_base(self, shape, gen=None):
        eps = torch.randn(shape, generator=gen)
        eps[..., 0] = torch.abs(eps[..., 0])
        return eps

    def forward(self, x, t, ijet):
        mv, s = self.embed_into_ga(x, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v = self.extract_from_ga(mv_outputs, s_outputs)
        return v

    def embed_into_ga(self, x, t, ijet):
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        mv = embed_vector(x).unsqueeze(-2)
        s = torch.cat([type_token, process_token, t_embedding], dim=2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        v[..., self.is_onshell, 3] = 0.0
        return v
