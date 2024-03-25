# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
# import dgl
import math
import torch
from torch import nn
from torch.autograd import grad

from torchdiffeq import odeint
from experiments.eventgen.transforms import ensure_angle
from experiments.eventgen.distributions import BaseDistribution, Naive4Momenta


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
        clamp_mse=None,
        hutchinson=True,
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_t_dim, scale=embed_t_scale),
            nn.Linear(embed_t_dim, embed_t_dim),
        )
        self.trace_fn = hutchinson_trace if hutchinson else autograd_trace

        if clamp_mse is not None:
            self.loss = lambda v1, v2: torch.mean(
                1
                / (
                    1 / nn.functional.mse_loss(v1, v2, reduction="none").clamp(max=1e10)
                    + 1 / clamp_mse
                )
            )
        else:
            self.loss = lambda v1, v2: nn.functional.mse_loss(v1, v2)

        # should be implemented by child classes
        self.distribution = BaseDistribution()

    def get_trajectory(self, x0, eps, t):
        # default: linear trajectory
        distance = eps - x0
        x_t = x0 + distance * t
        v_t = distance
        return x_t, v_t

    def batch_loss(self, x0, ijet):
        t = torch.rand(x0.shape[0], 1, 1, dtype=x0.dtype, device=x0.device)
        eps = self.distribution.sample(x0.shape).to(device=x0.device, dtype=x0.dtype)
        x_t, v_t = self.get_trajectory(x0, eps, t)

        v_pred = self.get_velocity(x_t, t, ijet=ijet)

        loss = self.loss(v_pred, v_t)
        return loss

    def sample(self, ijet, shape, device, dtype):
        def velocity(t, x_t):
            t = t * torch.ones(shape[0], 1, 1, dtype=dtype, device=device)
            v_t = self.get_velocity(x_t, t, ijet=ijet)
            return v_t

        eps = self.distribution.sample(shape).to(device=device, dtype=dtype)
        x_t = odeint(
            velocity,
            eps,
            torch.tensor([1.0, 0.0]),
            method="rk4",
            options={"step_size": 1e-2},
        )[-1]
        return x_t

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
            torch.tensor([0, 1], dtype=x.dtype, device=x.device),
            method="rk4",
            options={"step_size": 1e-2},
        )
        eps = x_t[-1].detach()
        jac = logp_diff[-1].detach()
        log_prob_base = self.distribution.log_prob(eps).sum(dim=[1, 2]).unsqueeze(-1)
        log_prob = log_prob_base - jac
        return log_prob


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    Save information at the wrapper level, have wrapper-specific preprocessing and undo_preprocessing
    """

    def __init__(self, *args):
        super().__init__(*args)

    def init_physics(self, units, pt_min, onshell_list, onshell_mass, delta_r_min):
        self.units = units
        self.pt_min = torch.tensor(pt_min).unsqueeze(0) / self.units
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.delta_r_min = delta_r_min

        # same preprocessing for all multiplicities
        self.prep_params = {}

        # base distribution
        self.distribution = Naive4Momenta(self.onshell_list, self.onshell_mass)

    def preprocess(self, fourmomenta):
        raise NotImplementedError

    def undo_preprocess(self, x):
        raise NotImplementedError
