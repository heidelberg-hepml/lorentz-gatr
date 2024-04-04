# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import numpy as np
import torch
from torch import nn
from torch.autograd import grad

from torchdiffeq import odeint
from experiments.eventgen.transforms import ensure_angle
from experiments.eventgen.distributions import (
    BaseDistribution,
    Distribution1,
    Distribution2,
)
from experiments.eventgen.coordinates import BaseCoordinates


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


def hutchinson_trace(x_out, x_in):
    # Hutchinson's trace Jacobian estimator, needs O(1) calls to autograd
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.0
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0].detach()
    return torch.sum(gradient * noise, dim=[1, 2])


def autograd_trace(x_out, x_in):
    # Standard way of calculating trace of the Jacobian, needs O(n) calls to autograd
    trJ = 0.0
    for i in range(x_out.shape[1]):
        for j in range(x_out.shape[2]):
            trJ += (
                grad(x_out[:, i, j].sum(), x_in, retain_graph=True)[0]
                .contiguous()[:, i, j]
                .contiguous()
                .detach()
            )
    return trJ.contiguous()


class CFM(nn.Module):
    """
    Base class for all CFM models
    - sample_base and get_velocity should be implemented by subclasses
    - get_trajectory, batch_loss, sample and log_prob might be overwritten or extended by subclasses
    """

    def __init__(
        self,
        embed_t_dim,
        embed_t_scale,
        odeint_kwargs={"method": "dopri5", "atol": 1e-9, "rtol": 1e-7, "method": None},
        hutchinson=True,
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_t_dim, scale=embed_t_scale),
            nn.Linear(embed_t_dim, embed_t_dim),
        )
        self.trace_fn = hutchinson_trace if hutchinson else autograd_trace
        self.odeint_kwargs = odeint_kwargs
        self.loss = lambda v1, v2: nn.functional.mse_loss(v1, v2)

    def init_distribution(self):
        self.distribution = BaseDistribution()

    def init_coordinates(self):
        self.coordinates = BaseCoordinates()

    def get_distance(self, x0, eps):
        return eps - x0

    def get_trajectory(self, x0, eps, t):
        # default: linear trajectory
        distance = self.get_distance(x0, eps)
        x_t = x0 + distance * t
        v_t = distance
        return x_t, v_t

    def sample_base(self, shape, device, dtype):
        fourmomenta = self.distribution.sample(shape).to(device=device, dtype=dtype)
        x = self.coordinates.fourmomenta_to_x(fourmomenta)
        return x

    def batch_loss(self, x0, ijet):
        t = torch.rand(x0.shape[0], 1, 1, dtype=x0.dtype, device=x0.device)
        eps = self.sample_base(x0.shape, x0.device, x0.dtype)
        x_t, v_t = self.get_trajectory(x0, eps, t)

        v_pred = self.get_velocity(x_t, t, ijet=ijet)

        loss = self.loss(v_pred, v_t)
        return loss

    def sample(
        self, ijet, shape, device, dtype, trajectory_path=None, n_trajectories=100
    ):
        # overhead for saving trajectories
        save_trajectory = trajectory_path != None
        if save_trajectory:
            xts, vts, ts = [], [], []

        def velocity(t, x_t):
            t = t * torch.ones(shape[0], 1, 1, dtype=x_t.dtype, device=x_t.device)
            v_t = self.get_velocity(x_t, t, ijet=ijet)

            # collect trajectories
            if save_trajectory:
                xts.append(x_t[:, :n_trajectories, ...])
                vts.append(v_t[:, :n_trajectories, ...])
                ts.append(t[0, 0, 0])
            return v_t

        eps = self.sample_base(shape, device, dtype)
        x = odeint(
            velocity,
            eps,
            torch.tensor([1.0, 0.0]),
            **self.odeint_kwargs,
        )[-1]

        # save trajectories to file
        if save_trajectory:
            # collect trajectories
            xts_learned = torch.stack(xts, dim=0)
            vts_learned = torch.stack(vts, dim=0)
            ts = torch.stack(ts, dim=0)
            xts_true, vts_true = self.get_trajectory(
                xts_learned[-1, ...]
                .reshape(1, *xts_learned.shape[1:])
                .expand(xts_learned.shape),
                xts_learned[0, ...]
                .reshape(1, *xts_learned.shape[1:])
                .expand(xts_learned.shape),
                ts.reshape(ts.shape[0], 1, 1, 1),
            )

            # transform to fourmomenta space
            xts_learned_fm = self.coordinates.x_to_fourmomenta(xts_learned)
            xts_true_fm = self.coordinates.x_to_fourmomenta(xts_true)
            vts_learned_fm = self.coordinates.velocities_x_to_fourmomenta(
                vts_learned, xts_learned, xts_learned_fm
            )
            vts_true_fm = self.coordinates.velocities_x_to_fourmomenta(
                vts_true, xts_true, xts_true_fm
            )

            # save
            np.savez_compressed(
                trajectory_path,
                xts_learned=xts_learned_fm.cpu() * self.units,
                vts_learned=vts_learned_fm.cpu() * self.units,
                xts_true=xts_true_fm.cpu() * self.units,
                vts_true=vts_true_fm.cpu() * self.units,
                ts=ts.cpu(),
            )

        # coordinate-specific checks
        x = self.coordinates.final_checks(x)
        return x

    def get_velocity(self, x, t, ijet):
        raise NotImplementedError

    def log_prob(self, x, ijet):
        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                x_t = state[0].detach().requires_grad_(True)
                t = t * torch.ones(x.shape[0], 1, 1, dtype=x.dtype, device=x.device)
                v_t = self.get_velocity(x_t, t, ijet=ijet)
                dlogp_dt = self.trace_fn(v_t, x_t).unsqueeze(-1)
            return v_t.detach(), dlogp_dt.detach()

        logp_diff = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        state = (x, logp_diff)
        x_t, logp_diff = odeint(
            net_wrapper,
            state,
            torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device),
            **self.odeint_kwargs,
        )
        eps = x_t[-1].detach()
        jac = logp_diff[-1].detach()
        log_prob_base = self.distribution.log_prob(eps).unsqueeze(-1)
        log_prob = log_prob_base + jac
        return log_prob


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    Save information at the wrapper level, have wrapper-specific preprocessing and undo_preprocessing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_physics(
        self,
        units,
        pt_min,
        delta_r_min,
        onshell_list,
        onshell_mass,
        base_kwargs,
        base_type,
        use_pt_min,
        use_delta_r_min,
        mass_scale,
    ):
        self.units = units
        self.pt_min = pt_min
        self.delta_r_min = delta_r_min
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.base_kwargs = base_kwargs
        self.base_type = base_type
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min
        self.mass_scale = mass_scale

        # same preprocessing for all multiplicities
        self.prep_params = {}

    def init_distribution(self):
        # simple base distribution
        if self.base_type == 1:
            self.distribution = Distribution1(
                self.onshell_list,
                self.onshell_mass,
                self.units,
                self.base_kwargs,
                self.delta_r_min,
                self.pt_min,
                self.use_delta_r_min,
                self.use_pt_min,
            )
        elif self.base_type == 2:
            self.distribution = Distribution2(
                self.onshell_list,
                self.onshell_mass,
                self.units,
                self.base_kwargs,
                self.delta_r_min,
                self.pt_min,
                self.use_delta_r_min,
                self.use_pt_min,
            )
        else:
            raise ValueError(f"base_type={self.base_type} not implemented")

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        x = self.coordinates.fourmomenta_to_x(fourmomenta)
        return x

    def undo_preprocess(self, x):
        fourmomenta = self.coordinates.x_to_fourmomenta(x)
        fourmomenta = fourmomenta * self.units
        return fourmomenta
