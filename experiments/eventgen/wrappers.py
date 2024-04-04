# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import torch
from torch import nn
from torchdiffeq import odeint

from gatr.interface import embed_vector, extract_vector, extract_scalar
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.toptagging.dataset import embed_beam_reference
from experiments.eventgen.cfm import EventCFM
from experiments.eventgen.transforms import (
    ensure_angle,
    delta_r,
)
import experiments.eventgen.distributions as distributions
import experiments.eventgen.coordinates as coordinates


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


def distance_phi(x0, eps):
    # take into account that phi is cyclic when computing shortest distance
    distance = eps - x0
    distance[..., 1] = ensure_angle(distance[..., 1])
    return distance


### CFM on 4momenta


class MLPCFMFourmomenta(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net

    def init_coordinates(self):
        self.coordinates = coordinates.Fourmomenta()

    def get_velocity(self, x, t, ijet):
        t_embedding = self.t_embedding(t).squeeze()
        x = x.reshape(x.shape[0], -1)

        x = torch.cat([x, t_embedding], dim=-1)
        v = self.net(x)
        v = v.reshape(v.shape[0], v.shape[1] // 4, 4)
        return v


class GAPCFMFourmomenta(EventCFM):
    def __init__(
        self,
        net,
        embed_t_dim,
        embed_t_scale,
        beam_reference,
        add_time_reference,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def init_coordinates(self):
        self.coordinates = coordinates.Fourmomenta()

    def get_velocity(self, fourmomenta, t, ijet):
        # GATr in fourmomenta space
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta

    def embed_into_ga(self, x, t, ijet):
        # note: ijet is not used
        # (joint training only supported for transformers)

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
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels

    def get_velocity(self, x, t, ijet):
        # note: flow matching happens directly in x space
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x = torch.cat([x, type_token, process_token, t_embedding], dim=-1)
        v = self.net(x)
        return v


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
        hutchinson=True,
    ):
        super().__init__(
            embed_t_dim,
            embed_t_scale,
            odeint_kwargs=odeint_kwargs,
            hutchinson=hutchinson,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.beam_reference = beam_reference
        self.add_time_reference = add_time_reference

    def get_velocity(self, x, t, ijet):
        x = self.coordinates.final_checks(x)
        fourmomenta = self.coordinates.x_to_fourmomenta(x)

        # GATr in fourmomenta space
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)

        v_x = self.coordinates.velocities_fourmomenta_to_x(
            v_fourmomenta, fourmomenta, x
        )
        return v_x

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


# Note: For now, have a seperate class for every coordinate set (to make it easier to hack things)
# Later, either pick one coordinate set or have a switch for it


class TransformerCFMFourmomenta(TransformerCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Fourmomenta()


class TransformerCFMJetmomenta(TransformerCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Jetmomenta()

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class TransformerCFMPrecisesiast(TransformerCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Precisesiast(self.pt_min, self.units)

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class GATrCFMFourmomenta(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Fourmomenta()


class GATrCFMPtPhiEtaE(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.PtPhiEtaE()

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class GATrCFMPPPM(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.PPPM(mass_scale=self.mass_scale)


class GATrCFMPPPM2(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.PPPM2(mass_scale=self.mass_scale)


class GATrCFMPPPlogM(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.PPPlogM(mass_scale=self.mass_scale)


class GATrCFMPPPlogM2(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.PPPlogM2(mass_scale=self.mass_scale)


class GATrCFMJetmomenta(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Jetmomenta(mass_scale=self.mass_scale)

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class GATrCFMJetmomenta2(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Jetmomenta2(mass_scale=self.mass_scale)

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class GATrCFMPrecisesiast(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Precisesiast(
            self.pt_min, self.units, mass_scale=self.mass_scale
        )

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


class GATrCFMPrecisesiast2(GATrCFM):
    def init_coordinates(self):
        self.coordinates = coordinates.Precisesiast2(
            self.pt_min, self.units, mass_scale=self.mass_scale
        )

    def get_distance(self, x0, eps):
        return distance_phi(x0, eps)


# deltaR business


class TransformerCFMPrecisesiastDeltaR(TransformerCFMPrecisesiast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 1.0
        self.k = 0.5

    def get_trajectory(self, x0, eps, t):
        # naive distance
        distance_naive = eps - x0
        distance_naive[..., 1] = ensure_angle(distance_naive[..., 1])

        # set naive distance (will overwrite the [..., [1,2]] components later)
        x_t = x0 + distance_naive * t
        v_t = distance

        def potential(diff):
            return (diff[..., 1] ** 2 + diff[..., 2] ** 2 - self.delta_r_min**2) ** (
                -self.k
            )

        def get_diff(x, i, j):
            diff = x[..., i, :] - x[..., j, :]
            diff[..., 0] = ensure_angle(diff[..., 0])
            return diff

        def potential_metric(x0, x1):
            terms = 0.0
            for i in range(x0.shape[1]):
                for j in range(i):
                    diff_x0, diff_x1 = diff(x0, i, j), diff(x1, i, j)
                    terms += (potential(diff_x0) - potential(diff_x1)) ** 2
            return self.alpha * terms

        def velocity(ta, x_t):
            # Fixed velocity function
            # Note: ta is not the same as t; both are not used here
            metric = torch.sum(
                distance_naive[..., [1, 2]] ** 2, dim=-2
            ) + potential_metric(x0, x_t)
            gradient_phi = 0.0  # TBD
            gradient_eta = 0.0  # TBD
            v_t = (
                metric
                * torch.stack([gradient_phi, gradient_eta], dim=-1)
                / torch.sqrt(gradient_phi**2 + gradient_eta**2)
            )
            return v_t

        x_t = odeint(velocity, eps, torch.tensor([1.0, t]), **self.odeint_kwargs)[-1]
        v_t = get_velocity(None, x_t)
        return x_t, v_t
