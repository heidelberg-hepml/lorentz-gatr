import torch
import math

from experiments.eventgen.transforms import (
    EPS1,
    CUTOFF,
    get_mass,
    get_pt,
    delta_r,
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
)

# sample a few extra events to speed up rejection sampling
SAMPLING_FACTOR = 100


class BaseDistribution:
    """
    Abstract base distribution
    All child classes work in fourmomenta space,
    i.e. they generate fourmomenta and return log_prob in fourmomenta space
    """

    def __init__(self):
        pass

    def sample(self, shape, generator=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError


class Distribution(BaseDistribution):
    """
    Base class for distributions
    Has rejection sampling based on pt and delta_r
    """

    def __init__(
        self,
        onshell_list,
        onshell_mass,
        units,
        delta_r_min,
        pt_min,
        use_delta_r_min,
        use_pt_min,
    ):
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.units = units

        self.delta_r_min = delta_r_min
        self.pt_min = torch.tensor(pt_min)
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min

    def propose(self, shape, generator=None):
        raise NotImplementedError

    def sample(self, shape, generator=None):
        def collect():
            fourmomenta = self.propose(shape, generator=generator)
            mask = torch.ones_like(fourmomenta[:, 0, 0], dtype=torch.bool)
            if self.use_pt_min:
                pt_mask = get_pt_mask(fourmomenta * self.units, self.pt_min)
                mask *= pt_mask
            if self.use_delta_r_min:
                delta_r_mask = get_delta_r_mask(
                    fourmomenta * self.units, self.delta_r_min
                )
                mask *= delta_r_mask
            fourmomenta = fourmomenta[mask, ...]
            return fourmomenta

        # collect events
        fourmomenta = []
        nevents = 0
        while nevents < shape[0]:
            fourmomenta_new = collect()
            fourmomenta.append(fourmomenta_new)
            nevents = sum([x.shape[0] for x in fourmomenta])

        # wrap up
        fourmomenta = torch.cat(fourmomenta, dim=0)
        fourmomenta = fourmomenta[: shape[0], ...]
        return fourmomenta

    def get_efficiency_factor(self, shape):
        """
        Effective volume of space that we sample from
        (i.e. not rejected because of pt_min or delta_r_min)
        Should rescale probability by 1/efficiency
        """
        fourmomenta = self.propose(shape)
        mask = torch.ones_like(fourmomenta[:, 0, 0], dtype=torch.bool)
        if self.use_pt_min:
            pt_mask = get_pt_mask(fourmomenta * self.units, self.pt_min)
            mask *= pt_mask
        if self.use_delta_r_min:
            delta_r_mask = get_delta_r_mask(fourmomenta * self.units, self.delta_r_min)
            mask *= delta_r_mask
        efficiency = mask.float().mean()
        return efficiency

    def log_prob(self, fourmomenta):
        log_prob_raw = self.log_prob_raw(fourmomenta)

        # include efficiency effect
        efficiency = self.get_efficiency_factor(fourmomenta.shape)
        log_prob = log_prob_raw - math.log(efficiency)
        return log_prob


class Distribution1(Distribution):
    """Base distribution 1: 3-momentum from fitted normal, mass from fitted log-normal"""

    def __init__(
        self,
        onshell_list,
        onshell_mass,
        units,
        base_kwargs,
        delta_r_min,
        pt_min,
        use_delta_r_min,
        use_pt_min,
    ):
        super().__init__(
            onshell_list,
            onshell_mass,
            units,
            delta_r_min,
            pt_min,
            use_delta_r_min,
            use_pt_min,
        )

        self.pxy_std = base_kwargs["pxy_std"]
        self.pz_std = base_kwargs["pz_std"]
        self.logmass_mean = base_kwargs["logmass_mean"]
        self.logmass_std = base_kwargs["logmass_std"]

    def propose(self, shape, generator=None):
        # sample (logmass, px, py, pz)
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, generator=generator)
        px = eps[..., 1] * self.pxy_std
        py = eps[..., 2] * self.pxy_std
        pz = eps[..., 3] * self.pz_std
        logmass = eps[..., 0] * self.logmass_std + self.logmass_mean

        # construct mass
        mass = logmass.exp() - EPS1
        mass[..., self.onshell_list] = (
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
        )

        # recover fourmomenta
        E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)
        fourmomenta = torch.stack((E, px, py, pz), dim=-1) / self.units
        assert torch.isfinite(fourmomenta).all()
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        fourmomenta *= self.units
        log_prob = torch.zeros_like(fourmomenta)
        log_prob[..., [1, 2]] = log_prob_normal(
            fourmomenta[..., [1, 2]], std=self.pxy_std
        )
        log_prob[..., 3] = log_prob_normal(fourmomenta[..., 3], std=self.pz_std)

        # special treatment for the mass
        mass = get_mass(fourmomenta)
        log_prob_logmass = log_prob_normal(
            (mass + EPS1).log(), mean=self.logmass_mean, std=self.logmass_std
        )
        log_prob_logmass[
            ..., self.onshell_list
        ] = 0.0  # no contribution from fixed masses
        log_prob_mass = log_prob_logmass / mass
        log_prob[..., 0] = log_prob_mass * fourmomenta[..., 0] / mass
        log_prob = log_prob.sum(dim=[1, 2])
        assert torch.isfinite(log_prob).all()
        return log_prob


class Distribution2(Distribution):
    """Base distribution 1: phi uniform; eta, log(pt), log(mass) from fitted normal"""

    def __init__(
        self,
        onshell_list,
        onshell_mass,
        units,
        base_kwargs,
        delta_r_min,
        pt_min,
        use_delta_r_min,
        use_pt_min,
    ):
        super().__init__(
            onshell_list,
            onshell_mass,
            units,
            delta_r_min,
            pt_min,
            use_delta_r_min,
            use_pt_min,
        )
        assert use_pt_min, f"use_pt_min=False not implemented for Distribution2"

        # construct mean and std on log(pt-pt_min) from mean and std on log(pt)
        logpt_mean = torch.tensor([base_kwargs["logpt_mean"]])
        logpt_std = torch.tensor([base_kwargs["logpt_std"]])
        self.logpt_mean = torch.log(logpt_mean.exp() - self.pt_min - EPS1)
        self.logpt_std = torch.log(
            (torch.exp(logpt_mean + logpt_std) - self.pt_min - EPS1)
            / (torch.exp(logpt_mean) - self.pt_min - EPS1)
        )

        self.logmass_mean = base_kwargs["logmass_mean"]
        self.logmass_std = base_kwargs["logmass_std"]
        self.eta_std = base_kwargs["eta_std"]

    def propose(self, shape, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        # sample (logpt, phi, eta, logmass)
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, generator=generator)
        eta = eps[..., 2] * self.eta_std
        logmass = eps[..., 3] * self.logmass_std + self.logmass_mean
        phi = math.pi * (2 * torch.rand(shape[:-1], generator=generator) - 1)

        # construct pt
        logpt = eps[..., 0] * self.logpt_std[: shape[1]] + self.logpt_mean[: shape[1]]
        pt = logpt.exp() + self.pt_min[: shape[1]] - EPS1

        # construct mass
        mass = logmass.exp() - EPS1
        mass[..., self.onshell_list] = (
            torch.tensor(self.onshell_mass)
            .unsqueeze(0)
            .expand(shape[0], len(self.onshell_list))
        )

        # convert to fourmomenta
        jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta) / self.units
        assert torch.isfinite(fourmomenta).all()
        return fourmomenta

    def log_prob(self, fourmomenta):
        fourmomenta *= self.units
        log_prob = torch.zeros_like(fourmomenta)

        # compute raw log_prob's
        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        E, px, py, pz = torch.permute(fourmomenta, (2, 0, 1))
        pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))
        log_prob_logpt = log_prob_normal(
            (pt - self.pt_min[: fourmomenta.shape[1]] + EPS1).log(),
            mean=self.logpt_mean[: fourmomenta.shape[1]],
            std=self.logpt_std[: fourmomenta.shape[1]],
        )
        log_prob_phi = -math.log(2 * math.pi) * torch.ones_like(log_prob_logpt)
        log_prob_eta = log_prob_normal(eta, std=self.eta_std)
        log_prob_logmass = log_prob_normal(
            (mass + EPS1).log(), mean=self.logmass_mean, std=self.logmass_std
        )
        log_prob_logmass[..., self.onshell_list] = 0.0

        # collect log_prob in jetmomenta space
        log_prob_mass = log_prob_logmass / (mass + EPS1)
        log_prob_pt = log_prob_logpt / (pt - self.pt_min[: fourmomenta.shape[1]] + EPS1)
        log_prob_jet = torch.stack(
            (log_prob_pt, log_prob_phi, log_prob_eta, log_prob_mass), dim=-1
        )

        log_prob_jet = log_prob_jet.sum(dim=[1, 2])
        assert torch.isfinite(log_prob_jet).all()
        # transform log_prob's to fourmomenta
        jac = (
            mass * pt**2 * torch.cosh(eta.clamp(min=-CUTOFF, max=CUTOFF)) / E
        )  # dfourmomenta/djetmomenta
        log_prob = log_prob_jet - jac.log().sum(dim=-1)
        assert torch.isfinite(log_prob).all()
        return log_prob


def get_pt_mask(fourmomenta, pt_min):
    pt = get_pt(fourmomenta)
    pt_min = pt_min[: fourmomenta.shape[1]]
    mask = (pt > pt_min).all(dim=-1)
    return mask


def get_delta_r_mask(fourmomenta, delta_r_min):
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    n_particles = fourmomenta.shape[1]
    mask = torch.ones_like(fourmomenta[:, 0, 0], dtype=torch.bool)
    for idx1 in range(n_particles):
        for idx2 in range(idx1):
            dr = delta_r(jetmomenta, idx1, idx2)
            mask *= dr > delta_r_min
    return mask


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
