import torch
import math

from experiments.eventgen.helpers import (
    get_pt,
    delta_r_fast,
    unpack_last,
    fourmomenta_to_jetmomenta,
)
import experiments.eventgen.coordinates as c

# sample a few extra events to speed up rejection sampling
SAMPLING_FACTOR = 10  # typically acceptance_rate > 0.5

from experiments.eventgen.helpers import EPS1


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


class RejectionDistribution(BaseDistribution):
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
        The 'normalization' factor is estimated using the acceptance rate of rejection sampling
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


class StandardPPPM2(RejectionDistribution):
    """Base distribution 1: 3-momentum from standard normal, mass from standard half-normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PPPM2()

    def propose(self, shape, device, dtype, generator=None):
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        m2 = eps[..., 0].abs()
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        m2[..., self.onshell_list] = (onshell_mass / self.units) ** 2
        pppm2 = torch.cat((eps[..., 1:], m2.unsqueeze(-1)), dim=-1)
        fourmomenta = self.coordinates.x_to_fourmomenta(pppm2)
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        pppm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(pppm2)
        log_prob[..., 3] += math.log(2)  # normalization factor because half-gaussian
        log_prob[..., self.onshell_list, 3] = 0.0  # fixed components do not contribute
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(pppm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class StandardPPPLogM2(RejectionDistribution):
    """Base distribution 2: 3-momentum from standard normal, log(mass) from standard normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for 4-momenta: 3-momentum and log-mass from standard normal"""
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        eps[..., self.onshell_list, 3] = torch.log(
            (onshell_mass / self.units) ** 2 + EPS1
        )
        fourmomenta = self.coordinates.x_to_fourmomenta(eps)
        return fourmomenta

    def log_prob_raw(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ppplogm2)
        log_prob[..., self.onshell_list, 3] = 0.0
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ppplogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class FittedPPPLogM2(RejectionDistribution):
    """Base distribution 3: 3-momentum and mass from fitted normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = c.PPPLogM2()

    def propose(self, shape, device, dtype, generator=None):
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        # onshell business
        eps = self.coordinates.transforms[-1].inverse(eps)
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        eps[..., self.onshell_list, 3] = torch.log(
            (onshell_mass / self.units) ** 2 + EPS1
        )

        for t in self.coordinates.transforms[:-1][::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob_raw(self, fourmomenta):
        ppplogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(ppplogm2)
        log_prob[..., self.onshell_list, 3] = 0.0
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(ppplogm2)[0]
        log_prob = log_prob + logdetjac
        return log_prob


class FittedLogPtPhiEtaLogM2(RejectionDistribution):
    """Base distribution 4: phi uniform; eta, log(pt) and log(mass) from fitted normal"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.use_pt_min
        ), f"use_pt_min=False not implemented for distribution FittedLogPtPhiEtaLogM2"
        self.coordinates = c.LogPtPhiEtaLogM2(self.pt_min, self.units)

    def propose(self, shape, device, dtype, generator=None):
        """Base distribution for precisesiast: pt, eta gaussian; phi uniform; mass shifted gaussian"""
        # sample (logpt, phi, eta, logmass)
        shape = list(shape)
        shape[0] = int(shape[0] * SAMPLING_FACTOR)
        eps = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        # onshell business
        eps = self.coordinates.transforms[-1].inverse(eps)
        onshell_mass = self.onshell_mass.to(device, dtype=dtype).unsqueeze(0)
        eps[..., self.onshell_list, 3] = torch.log(
            (onshell_mass / self.units) ** 2 + EPS1
        )

        # be careful with phi and eta
        eps[..., 1] = math.pi * (
            2 * torch.rand(shape[:-1], device=device, dtype=dtype, generator=generator)
            - 1
        )  # sample phi uniformly

        for t in self.coordinates.transforms[:-1][::-1]:
            eps = t.inverse(eps)
        return eps

    def log_prob_raw(self, fourmomenta):
        logptphietalogm2 = self.coordinates.fourmomenta_to_x(fourmomenta)
        log_prob = log_prob_normal(logptphietalogm2)
        log_prob[..., 1] = -math.log(
            2 * math.pi
        )  # normalization factor for uniform phi distribution: 1/(2 pi)
        log_prob[..., self.onshell_list, 3] = 0.0
        log_prob = log_prob.sum(dim=[-1, -2]).unsqueeze(-1)
        logdetjac = self.coordinates.logdetjac_x_to_fourmomenta(logptphietalogm2)[0]
        log_prob = log_prob + logdetjac
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
