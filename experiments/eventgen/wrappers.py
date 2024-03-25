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
    EPS1,
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


# Note: Should eventually have seperate class for base distributions
# (with sample() and log_prob() methods)
def base_4momenta(shape, onshell_list, onshell_mass, generator=None):
    """Base distribution for 4-momenta: 3-momentum from standard gaussian, mass from half-gaussian"""
    eps = torch.randn(shape, generator=generator)
    mass = eps[..., 0].abs()
    mass[..., onshell_list] = torch.log(
        torch.tensor(onshell_mass).unsqueeze(0).expand(shape[0], len(onshell_list))
        + EPS1
    )
    eps[..., 0] = torch.sqrt(mass**2 + torch.sum(eps[..., 1:] ** 2, dim=-1))
    assert torch.isfinite(eps).all()
    return eps


def base_precisesiast(
    shape,
    onshell_list,
    onshell_mass,
    delta_r_min,
    generator=None,
    std_pt=0.8,
    mean_pt=-1,
    std_mass=0.5,
    mean_mass=-3,
    std_eta=1.5,
):
    """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
    pt = torch.randn(shape[:-1], generator=generator) * std_pt + mean_pt
    if delta_r_min is None:
        eta = torch.randn(shape[:-1], generator=generator) * std_eta
        phi = math.pi * (2 * torch.rand(shape[:-1], generator=generator) - 1)
    else:
        phi, eta = eta_phi_no_deltar_holes(
            shape,
            generator=generator,
            delta_r_min=delta_r_min,
            std_eta=std_eta,
        )
    mass = torch.randn(shape[:-1], generator=generator) * std_mass + mean_mass
    mass[..., onshell_list] = torch.log(
        torch.tensor(onshell_mass).unsqueeze(0).expand(shape[0], len(onshell_list))
        + EPS1
    )
    eps = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(eps).all()
    return eps


def eta_phi_no_deltar_holes(
    shape, generator=None, delta_r_min=0.4, safety_factor=10, std_eta=1.0
):
    """Use rejection sampling to sample phi and eta based on 'shape' with delta_r > delta_r_min"""
    phi = math.pi * (
        2 * torch.rand(safety_factor * shape[0], shape[1], generator=generator) - 1
    )
    eta = torch.randn(safety_factor * shape[0], shape[1], generator=generator) * std_eta
    event = torch.stack(
        (torch.zeros_like(phi), phi, eta, torch.zeros_like(phi)), dim=-1
    )

    mask = []
    for idx1 in range(shape[1]):
        for idx2 in range(shape[1]):
            if idx1 >= idx2:
                continue
            mask_single = delta_r(event, idx1, idx2)
            mask.append(mask_single > delta_r_min)
    mask = torch.stack(mask, dim=-1).all(dim=-1)
    assert mask.sum() > shape[0], (
        f"Have mask.sum={mask.sum()} and shape[0]={shape[0]} "
        f"-> Should increase the safety_factor={safety_factor}"
    )
    event = event[mask, ...][: shape[0], ...]
    _, phi, eta, _ = torch.permute(event, (2, 0, 1))
    return phi, eta


class MLPCFM4Momenta(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        clamp_mse,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse,
        )
        self.net = net

    def preprocess(self, fourmomenta):
        return fourmomenta / self.units

    def undo_preprocess(self, fourmomenta):
        return fourmomenta * self.units

    def sample_base(self, shape, gen=None):
        # use precisesiast base density (has numerical problems)
        # precisesiast = base_precisesiast(
        #    shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        # )
        # jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        # fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = base_4momenta(
            shape, self.onshell_list, self.onshell_mass, generator=gen
        )
        return fourmomenta

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
        clamp_mse,
        beam_reference,
        add_time_reference,
    ):
        super().__init__(embed_t_dim, embed_t_scale, clamp_mse)
        self.net = net
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def preprocess(self, fourmomenta):
        return fourmomenta / self.units

    def undo_preprocess(self, fourmomenta):
        return fourmomenta * self.units

    def sample_base(self, shape, gen=None):
        # use precisesiast base density (has numerical problems)
        # precisesiast = base_precisesiast(
        #    shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        # )
        # jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        # fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = base_4momenta(
            shape, self.onshell_list, self.onshell_mass, generator=gen
        )
        return fourmomenta
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
        clamp_mse,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse,
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

    def sample_base(self, shape, gen=None):
        # use precisesiast base density (has numerical problems)
        # precisesiast = base_precisesiast(
        #    shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        # )
        # jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        # fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = base_4momenta(
            shape, self.onshell_list, self.onshell_mass, generator=gen
        )
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

    def sample_base(self, shape, gen=None):
        return base_precisesiast(
            shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        )

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
        clamp_mse,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            clamp_mse,
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

    def sample_base(self, shape, gen=None):
        # use precisesiast base density (has numerical problems)
        # precisesiast = base_precisesiast(
        #    shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        # )
        # jetmomenta = precisesiast_to_jetmomenta(precisesiast, self.pt_min)
        # fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        fourmomenta = base_4momenta(
            shape, self.onshell_list, self.onshell_mass, generator=gen
        )
        return fourmomenta


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

    def sample_base(self, shape, gen=None):
        return base_precisesiast(
            shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        )

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

    def sample_base(self, shape, gen=None):
        return base_precisesiast(
            shape, self.onshell_list, self.onshell_mass, self.delta_r_min, generator=gen
        )

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
