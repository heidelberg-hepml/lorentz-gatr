import torch
import math
from experiments.eventgen.transforms import EPS1


class BaseDistribution:
    def __init__(self):
        pass

    def sample(self, shape, generator=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError


class Distribution1(BaseDistribution):
    def __init__(self, onshell_list, onshell_mass, units, std_mass=1.0):
        self.onshell_list = onshell_list
        self.onshell_mass = [m / units for m in onshell_mass]
        self.units = units
        self.std_mass = std_mass / units

    def sample(self, shape, generator=None):
        """Base distribution for 4-momenta: 3-momentum from standard gaussian, mass from half-gaussian"""
        eps = torch.randn(shape, generator=generator)
        mass = eps[..., 0].abs() * self.std_mass
        mass[..., self.onshell_list] = torch.log(
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
            + EPS1
        )
        eps[..., 0] = torch.sqrt(mass**2 + torch.sum(eps[..., 1:] ** 2, dim=-1))
        assert torch.isfinite(eps).all()
        return eps

    def log_prob(self, x):
        log_prob = torch.zeros_like(x)
        log_prob[..., 1:] = log_prob_gauss(x[..., 1:])

        # special treatment for the mass
        mass = torch.sqrt(
            torch.clamp(x[..., 0] ** 2 - torch.sum(x[..., 1:] ** 2, dim=-1), min=1e-10)
        )
        log_prob_mass = 2 * log_prob_gauss(
            mass, std=self.std_mass
        )  # factor 2 because have half-gaussian
        log_prob_mass[..., self.onshell_list] = 0.0  # no contribution from fixed masses
        log_prob[..., 0] = log_prob_mass * x[..., 0] / mass  # p(E) = p(m) * dm/dE
        return log_prob


class Base2(BaseDistribution):
    def __init__(
        self,
        onshell_list,
        onshell_mass,
        units,
        delta_r_min,
        std_pt=0.8,
        mean_pt=-1,
        std_mass=0.5,
        mean_mass=-3,
        std_eta=1.5,
    ):
        self.onshell_list = onshell_list
        self.onshell_mass = [m / units for m in onshell_mass]
        self.units = units
        self.delta_r_min = delta_r_min

        self.std_pt = std_pt / units
        self.mean_pt = mean_pt / units
        self.std_mass = std_mass / units
        self.mean_mass = mean_mass / units
        self.std_eta = std_eta

    def sample(self, shape, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        pt = torch.randn(shape[:-1], generator=generator) * self.std_pt + self.mean_pt
        if self.delta_r_min is None:
            eta = torch.randn(shape[:-1], generator=generator) * self.std_eta
            phi = math.pi * (2 * torch.rand(shape[:-1], generator=generator) - 1)
        else:
            phi, eta = eta_phi_no_deltar_holes(
                shape,
                generator=generator,
                delta_r_min=self.delta_r_min,
                std_eta=self.std_eta,
            )
        mass = (
            torch.randn(shape[:-1], generator=generator) * self.std_mass
            + self.mean_mass
        )
        mass[..., self.onshell_list] = torch.log(
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
            + EPS1
        )
        eps = torch.stack((pt, phi, eta, mass), dim=-1)
        assert torch.isfinite(eps).all()
        return eps

    def log_prob(self, x):
        # this is non-trivial
        raise NotImplementedError


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


def log_prob_gauss(z, mean=0.0, std=1.0):
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std
