# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import torch
from torch import nn

from gatr.interface import embed_vector, extract_vector
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.toptagging.dataset import embed_beam_reference
from experiments.eventgen.cfm import CFM
from experiments.eventgen.transforms import (
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
    jetmomenta_to_precisesiast,
    precisesiast_to_jetmomenta,
    velocities_fourmomenta_to_jetmomenta,
    velocities_jetmomenta_to_precisesiast,
    stable_arctanh,
    ensure_angle,
    EPS1,
    EPS2,
)


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


class TrCFMWrapper(CFM):
    def __init__(
        self,
        net,
        onshell_list,
        onshell_mass,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
        )
        self.net = net
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels

    def sample_base(self, shape, gen=None):
        assert shape[-1] == 4
        eps = torch.randn(shape, generator=gen)
        eps[..., 1] = math.pi * (2 * torch.rand(*shape[:-1], generator=gen) - 1)
        eps[..., 3] -= 3.0  # masses are typically smaller
        eps[..., self.onshell_list, 3] = torch.log(
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
            + EPS1
        )
        assert torch.isfinite(
            eps
        ).all(), f"{torch.isnan(eps).sum(dim=[0,1])} {torch.isinf(eps).sum(dim=[0,1])}"
        return eps

    def log_prob_base(self, eps):
        raise NotImplementedError

    def get_velocity(self, x, t, ijet):
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x[..., 1] = ensure_angle(x[..., 1])
        x = torch.cat([x, type_token, process_token, t_embedding], dim=-1)

        v = self.net(x)
        v[..., self.onshell_list, 3] = 0.0  # dont generate masses for now
        return v


class GATrCFMWrapper(CFM):
    def __init__(
        self,
        net,
        onshell_list,
        onshell_mass,
        pt_min,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
        beam_reference,
        add_time_reference,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
        )
        self.net = net
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.pt_min = torch.tensor(pt_min).unsqueeze(0)
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def sample_base(self, shape, gen=None):
        assert shape[-1] == 4
        eps = torch.randn(shape, generator=gen)
        eps[..., 1] = math.pi * (2 * torch.rand(*shape[:-1], generator=gen) - 1)
        eps[..., 3] -= 3.0  # masses are typically smaller
        eps[..., self.onshell_list, 3] = torch.log(
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
            + EPS1
        )
        assert torch.isfinite(
            eps
        ).all(), f"{torch.isnan(eps).sum(dim=[0,1])} {torch.isinf(eps).sum(dim=[0,1])}"
        return eps

    def log_prob_base(self, eps):
        raise NotImplementedError

    def get_velocity(self, precisesiast, t, ijet):
        # this is not necessary, because the transform handles it
        precisesiast[..., 1] = ensure_angle(precisesiast[..., 1])

        # precisesiast -> fourmomenta
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

        # GATr in fourmomenta space
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)

        # velocities fourmomenta -> velocities precisesiast
        v_jetmomenta = velocities_fourmomenta_to_jetmomenta(
            v_fourmomenta, fourmomenta, jetmomenta
        )
        v_precisesiast = velocities_jetmomenta_to_precisesiast(
            v_jetmomenta, jetmomenta, precisesiast
        )

        # set mass-velocities to zero for on-shell particles
        v_precisesiast[..., self.onshell_list, 3] = 0.0
        return v_precisesiast

    def embed_into_ga(self, x, t, ijet):
        # scalar embedding
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        s = torch.cat([type_token, process_token, t_embedding], dim=-1)

        # mv embedding
        mv = embed_vector(x).unsqueeze(-2)
        beam = embed_beam_reference(mv, self.beam_reference, self.add_time_reference)
        if beam is not None:
            beam = beam.unsqueeze(1).expand(*mv.shape[:-2], beam.shape[-2], 16)
            mv = torch.cat([mv, beam], dim=-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v
