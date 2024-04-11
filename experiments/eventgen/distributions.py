import torch
import math

from experiments.eventgen.helpers import (
    EPS1,
    get_pt,
    delta_r_fast,
    unpack_last,
    fourmomenta_to_jetmomenta,
)
from experiments.eventgen.coordinates import PPPM2, PPPLogM2, LogPtPhiEtaLogM2

# sample a few extra events to speed up rejection sampling
SAMPLING_FACTOR = 10  # typically acceptance_rate > 0.5


class BaseDistribution:
    """
    Abstract base distribution
    All child classes work in fourmomenta space,
    i.e. they generate fourmomenta and return log_prob in fourmomenta space
    """

    def sample(self, shape, device, dtype, generator=None, **kwargs):
        raise NotImplementedError

    def log_prob(self, x, **kwargs):
        raise NotImplementedError


class Distribution(BaseDistribution):
    """
    Implement rejection sampling based on delta_r and pt
    This class is still abstract,
    the 'propose' and 'log_prob_raw' methods have to be implemented by subclasses
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
        self.onshell_mass = torch.tensor(onshell_mass)
        self.units = units

        self.delta_r_min = delta_r_min
        self.pt_min = torch.tensor(pt_min)
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min

    def propose(self, shape, device, dtype, generator=None):
        raise NotImplementedError

    def create_cut_mask(self, fourmomenta):
        mask = torch.ones_like(fourmomenta[:, 0, 0], dtype=torch.bool)
        if self.use_pt_min:
            pt_mask = get_pt_mask(fourmomenta * self.units, self.pt_min)
            mask *= pt_mask
        if self.use_delta_r_min:
            delta_r_mask = get_delta_r_mask(fourmomenta * self.units, self.delta_r_min)
            mask *= delta_r_mask
        return mask

    def sample(self, shape, device, dtype, generator=None):
        def collect():
            fourmomenta = self.propose(
                shape, device=device, dtype=dtype, generator=generator
            )
            mask = self.create_cut_mask(fourmomenta)
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

    def get_normalization_factor(self, shape, device, dtype, generator=None):
        """
        Take effective volume of space that we sample from into account
        (i.e. not rejected because of pt_min or delta_r_min)

        We do p_actual = 1/normalization * p_naive,
        where p_naive is only normalized when also defined on the rejected regions
        The 'normalization' factor is estimated by the acceptance rate of rejection sampling
        """
        fourmomenta = self.propose(shape, device, dtype, generator=None)
        mask = self.create_cut_mask(fourmomenta)
        normalization = mask.float().mean()
        return normalization

    def log_prob(self, fourmomenta):
        log_prob_raw = self.log_prob_raw(fourmomenta)

        # include normalization effect
        normalization = self.get_normalization_factor(
            fourmomenta.shape, fourmomenta.device, fourmomenta.dtype
        )
        log_prob = log_prob_raw - math.log(normalization)
        return log_prob

    def log_prob_raw(self, fourmomenta):
        raise NotImplementedError


class NaiveDistribution(Distribution):
    """Base distribution 1: 3-momentum from standard normal, mass from standard half-normal"""

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
        self.m2_std = 1.0
        self.coordinates = PPPM2()

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for 4-momenta: 3-momentum from standard gaussian, mass from half-gaussian"""
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        m2 = eps[..., 0].abs() * self.m2_std
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        m2[..., self.onshell_list] = (onshell_mass / self.units) ** 2
        pppm2 = torch.cat((eps[..., 1:], m2.unsqueeze(-1)), dim=-1)
        fourmomenta = self.coordinates.x_to_fourmomenta(pppm2)
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        pppm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        p3, m2 = pppm2[..., :3], pppm2[..., 3]
        log_prob_p3 = log_prob_normal(p3)
        log_prob_m2 = log_prob_normal(m2, std=self.m2_std)
        log_prob_m2 += math.log(2)  # normalization factor because half-gaussian
        log_prob_m2[..., self.onshell_list] = 0.0  # fixed components do not contribute
        log_prob = torch.cat((log_prob_p3, log_prob_m2.unsqueeze(-1)), dim=-1)
        log_prob = log_prob.sum(dim=[-1, -2])
        log_prob = self.coordinates.log_prob_x_to_fourmomenta(log_prob, ppplogm2)
        return log_prob


class NaiveLogDistribution(Distribution):
    """Base distribution 1: 3-momentum from standard normal, mass from standard lognormal"""

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
        self.coordinates = PPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for 4-momenta: 3-momentum from standard gaussian, mass from half-gaussian"""
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        logm2 = eps[..., 3]
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        logm2[..., self.onshell_list] = torch.log(
            (onshell_mass / self.units) ** 2 + EPS1
        )
        ppplogm2 = torch.cat((eps[..., :3], logm2.unsqueeze(-1)), dim=-1)
        fourmomenta = self.coordinates.x_to_fourmomenta(ppplogm2)
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ppplogm2)
        log_prob[..., self.onshell_list, 3] = 0.0
        log_prob = log_prob.sum(dim=[-1, -2])
        log_prob = self.coordinates.log_prob_x_to_fourmomenta(log_prob, ppplogm2)
        return log_prob


class FourmomentaDistribution(Distribution):
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
        self.coordinates = PPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        px = eps[..., 1] * self.pxy_std
        py = eps[..., 2] * self.pxy_std
        pz = eps[..., 3] * self.pz_std
        logmass = eps[..., 0] * self.logmass_std + self.logmass_mean
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        logm2 = 2 * logmass
        logm2[..., self.onshell_list] = torch.log(onshell_mass**2 + EPS1)

        ppplogm2 = torch.stack((px, py, pz, logm2), dim=-1)
        fourmomenta = self.coordinates.x_to_fourmomenta(ppplogm2) / self.units
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta * self.units)
        px, py, pz, logm2 = unpack_last(ppplogm2)
        log_prob_px = log_prob_normal(px, std=self.pxy_std)
        log_prob_py = log_prob_normal(py, std=self.pxy_std)
        log_prob_pz = log_prob_normal(pz, std=self.pz_std)
        log_prob_logm2 = (
            log_prob_normal(logm2 / 2, mean=self.logmass_mean, std=self.logmass_std) / 2
        )
        log_prob_logm2[..., self.onshell_list] = 0.0
        log_prob = torch.stack(
            (log_prob_px, log_prob_py, log_prob_pz, log_prob_logm2), dim=-1
        )
        log_prob = log_prob.sum(dim=[-1, -2])
        log_prob = self.coordinates.log_prob_x_to_fourmomenta(log_prob, ppplogm2)
        return


class JetmomentaDistribution(Distribution):
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
        assert (
            use_pt_min
        ), f"use_pt_min=False not implemented for JetmomentaDistribution"
        self.coordinates = LogPtPhiEtaLogM2(pt_min, units)

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

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        # sample (logpt, phi, eta, logmass)
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        eta = eps[..., 2] * self.eta_std
        logmass = eps[..., 3] * self.logmass_std + self.logmass_mean
        logm2 = 2 * logmass
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        logm2[..., self.onshell_list] = torch.log(onshell_mass**2 + EPS1)
        u = torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
        phi = math.pi * (2 * u - 1)

        # construct pt
        logpt = eps[..., 0] * self.logpt_std[: shape[-2]].to(
            device, dtype=dtype
        ) + self.logpt_mean[: shape[1]].to(device, dtype=dtype)

        # convert to fourmomenta
        logptphietalogm2 = torch.stack((logpt, phi, eta, logm2), dim=-1)
        fourmomenta = self.coordinates.x_to_fourmomenta(logptphietalogm2) / self.units
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta * self.units)
        logpt, phi, eta, logm2 = unpack_last(logptphietalogm2)
        log_prob_eta = log_prob_normal(eta, std=self.eta_std)
        log_prob_phi = -math.log(2 * math.pi) * torch.ones_like(log_prob_eta)
        log_prob_logm2 = (
            log_prob_normal(logm2 / 2, mean=self.logmass_mean, std=self.logmass_std) / 2
        )  # careful with logm2 vs logm
        log_prob_logm2[..., self.onshell_list] = 0.0
        log_prob_logpt = log_prob_normal(
            (pt - self.pt_min[: fourmomenta.shape[1]] + EPS1).log(),
            mean=self.logpt_mean[: fourmomenta.shape[1]],
            std=self.logpt_std[: fourmomenta.shape[1]],
        )
        log_prob = torch.stack(
            (log_prob_logpt, log_prob_phi, log_prob_eta, log_prob_logm2), dim=-1
        )
        log_prob = log_prob.sum(dim=[-1, -2])
        log_prob = self.coordinates.log_prob_x_to_fourmomenta(
            log_prob, logptphietalogm2
        )
        return log_prob


def get_pt_mask(fourmomenta, pt_min):
    pt = get_pt(fourmomenta)
    pt_min = pt_min[: fourmomenta.shape[1]].to(
        fourmomenta.device, dtype=fourmomenta.dtype
    )
    mask = (pt > pt_min).all(dim=-1)
    return mask


def get_delta_r_mask(fourmomenta, delta_r_min):
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    dr = delta_r_fast(jetmomenta.unsqueeze(1), jetmomenta.unsqueeze(2))

    # diagonal should not be < delta_r_min
    arange = torch.arange(jetmomenta.shape[1], device=jetmomenta.device)
    dr[..., arange, arange] = 42

    mask = (dr > delta_r_min).all(dim=[-1, -2])
    return mask


def log_prob_normal(z, mean=0.0, std=1.0):
    std_term = torch.log(std) if type(std) == torch.Tensor else math.log(std)
    return -((z - mean) ** 2) / (2 * std**2) - 0.5 * math.log(2 * math.pi) - std_term
