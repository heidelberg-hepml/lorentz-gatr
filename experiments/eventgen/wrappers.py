# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import torch
from torch import nn

from gatr.interface import embed_vector, extract_vector, extract_scalar
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.toptagging.dataset import embed_beam_reference
from experiments.eventgen.cfm import EventCFM
from experiments.eventgen.transforms import (
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
    jetmomenta_to_precisesiast,
    precisesiast_to_jetmomenta,
    velocities_jetmomenta_to_fourmomenta,
    velocities_precisesiast_to_jetmomenta,
    velocities_jetmomenta_to_precisesiast,
    velocities_fourmomenta_to_jetmomenta,
    stable_arctanh,
    ensure_angle,
    delta_r,
)
from experiments.eventgen.distributions import FancyPrecisesiast


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


class MLPCFM4Momenta(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        clamp_mse=None,
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            odeint_kwargs=odeint_kwargs,
            clamp_mse=clamp_mse,
            hutchinson=hutchinson,
        )
        self.net = net

    def preprocess(self, fourmomenta):
        return fourmomenta / self.units

    def undo_preprocess(self, fourmomenta):
        return fourmomenta * self.units

    def get_velocity(self, x, t, ijet):
        t_embedding = self.t_embedding(t).squeeze()
        x = x.reshape(x.shape[0], -1)

        x = torch.cat([x, t_embedding], dim=-1)
        v = self.net(x)
        v = v.reshape(v.shape[0], v.shape[1] // 4, 4)
        return v


class GAPCFM4Momenta(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        beam_reference,
        add_time_reference,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        clamp_mse=None,
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse=clamp_mse,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def preprocess(self, fourmomenta):
        return fourmomenta / self.units

    def undo_preprocess(self, fourmomenta):
        return fourmomenta * self.units

    def get_velocity(self, fourmomenta, t, ijet):
        # GATr in fourmomenta space
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta

    def embed_into_ga(self, x, t, ijet):
        # scalar embedding
        s = self.t_embedding(t).squeeze()

        # mv embedding
        mv = embed_vector(x.reshape(x.shape[0], -1, 4))
        beam = embed_beam_reference(mv, self.beam_reference, self.add_time_reference)
        if beam is not None:
            mv = torch.cat([mv, beam], dim=-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v


class TransformerCFM(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        clamp_mse=None,
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse=clamp_mse,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels

    def get_velocity(self, x, t, ijet):
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x = torch.cat([x, type_token, process_token, t_embedding], dim=-1)
        v = self.net(x)
        return v


class TransformerCFM4Momenta(TransformerCFM):
    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        return fourmomenta

    def undo_preprocess(self, fourmomenta):
        fourmomenta = fourmomenta * self.units
        return fourmomenta


class TransformerCFMPrecisesiast(TransformerCFM):
    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        precisesiast = jetmomenta_to_precisesiast(jetmomenta, self.pt_min)
        return precisesiast

    def undo_preprocess(self, precisesiast):
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def get_velocity(self, x, t, ijet):
        x[..., 1] = ensure_angle(x[..., 1])
        v_precisesiast = super().get_velocity(x, t, ijet)
        v_precisesiast[..., self.onshell_list, 3] = 0.0
        return v_precisesiast

    def get_trajectory(self, x0, eps, t):
        distance = eps - x0
        distance[..., 1] = ensure_angle(distance[..., 1])
        x_t = x0 + distance * t
        v_t = distance
        return x_t, v_t

    def sample(self, *args):
        x_t = super().sample(*args)
        x_t[..., 1] = ensure_angle(x_t[..., 1])
        return x_t

    def init_physics(self, *args):
        super().init_physics(*args)
        self.distribution = FancyPrecisesiast(
            self.onshell_list, self.onshell_mass, self.delta_r_min
        )


class GATrCFM(EventCFM):
    """
    Abstract base class for all GATrCFM's
    Add GATr-specific parameters
    """

    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        type_token_channels,
        process_token_channels,
        beam_reference,
        add_time_reference,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        clamp_mse=None,
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse=clamp_mse,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def get_velocity(self, fourmomenta, t, ijet):
        # GATr in fourmomenta space
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta

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


class GATrCFM4Momenta(GATrCFM):
    """
    GATrCFM with flow matching in 4-momentum space
    The precisesiast representation does not appear at any point
    """

    def preprocess(self, fourmomenta):
        return fourmomenta / self.units

    def undo_preprocess(self, fourmomenta):
        return fourmomenta * self.units


class GATrCFMPrecisesiast1(GATrCFM):
    """
    GATrCFM with straight trajectories in precisesiast and CFM metric in fourmomenta
    """

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        precisesiast = jetmomenta_to_precisesiast(jetmomenta, self.pt_min)
        return precisesiast

    def undo_preprocess(self, precisesiast):
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def get_velocity(self, precisesiast, t, ijet):
        # this is not necessary, because the transform handles it
        precisesiast[..., 1] = ensure_angle(precisesiast[..., 1])

        # precisesiast -> fourmomenta
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

        v_fourmomenta = super().get_velocity(fourmomenta, t, ijet)
        return v_fourmomenta

    def get_trajectory(self, x0, eps, t):
        distance = eps - x0
        distance[..., 1] = ensure_angle(distance[..., 1])
        x_t = x0 + distance * t
        v_t = distance

        precisesiast = x_t
        v_precisesiast = v_t

        # onshell particles have velocity zero
        v_precisesiast[..., self.onshell_list, 3] = 0.0

        # transform velocity to fourmomenta space
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        v_jetmomenta = velocities_precisesiast_to_jetmomenta(
            v_precisesiast, precisesiast, jetmomenta
        )
        v_fourmomenta = velocities_jetmomenta_to_fourmomenta(
            v_jetmomenta, jetmomenta, fourmomenta
        )
        return precisesiast, v_fourmomenta

    def get_distance(self, x0, x1):
        distance = super().get_distance(x0, x1)
        distance[..., 1] = ensure_angle(distance[..., 1])
        return distance

    def sample(self, *args):
        x_t = super().sample(*args)
        x_t[..., 1] = ensure_angle(x_t[..., 1])
        return x_t

    def init_physics(self, *args):
        super().init_physics(*args)
        self.distribution = FancyPrecisesiast(
            self.onshell_list, self.onshell_mass, self.delta_r_min
        )


class GATrCFMPrecisesiast2(GATrCFM):
    """
    GATrCFM with straight trajectories in precisesiast and CFM metric in precisesiast
    """

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        precisesiast = jetmomenta_to_precisesiast(jetmomenta, self.pt_min)
        return precisesiast

    def undo_preprocess(self, precisesiast):
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def get_velocity(self, precisesiast, t, ijet):
        # this is not necessary, because the transform handles it
        precisesiast[..., 1] = ensure_angle(precisesiast[..., 1])

        # precisesiast -> fourmomenta
        jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

        v_fourmomenta = super().get_velocity(fourmomenta, t, ijet)

        v_jetmomenta = velocities_fourmomenta_to_jetmomenta(
            v_fourmomenta, fourmomenta, jetmomenta
        )
        v_precisesiast = velocities_jetmomenta_to_precisesiast(
            v_jetmomenta, jetmomenta, precisesiast, self.pt_min
        )
        v_precisesiast[..., self.onshell_list, 3] = 0.0
        return v_precisesiast

    def get_trajectory(self, x0, eps, t):
        distance = eps - x0
        distance[..., 1] = ensure_angle(distance[..., 1])
        x_t = x0 + distance * t
        v_t = distance

        # onshell particles have velocity zero
        v_t[..., self.onshell_list, 3] = 0.0
        return x_t, v_t

    def sample(self, *args):
        x_t = super().sample(*args)
        x_t[..., 1] = ensure_angle(x_t[..., 1])
        return x_t

    def init_physics(self, *args):
        super().init_physics(*args)
        self.distribution = FancyPrecisesiast(
            self.onshell_list, self.onshell_mass, self.delta_r_min
        )
