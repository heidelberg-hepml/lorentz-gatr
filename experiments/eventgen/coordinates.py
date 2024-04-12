import torch
import experiments.eventgen.transforms as tr

from experiments.eventgen.helpers import ensure_angle


class BaseCoordinates:
    """
    Class that implements transformations
    from fourmomenta to an abstract set of variables
    Heavily uses functionality from Transforms classes
    """

    def __init__(self):
        self.transforms = []

    def get_trajectory(self, x1, x2, t):
        v_t = x2 - x1
        x_t = x1 + t * v_t
        return x_t, v_t

    def fourmomenta_to_x(self, fourmomenta):
        x = fourmomenta.clone()
        for transform in self.transforms:
            x = transform.forward(x)
        return x

    def x_to_fourmomenta(self, x):
        x = x.clone()
        for transform in self.transforms[::-1]:
            x = transform.inverse(x)
        return x

    def velocity_fourmomenta_to_x(self, v_fourmomenta, fourmomenta):
        v = v_fourmomenta.clone()
        x = fourmomenta.clone()
        for transform in self.transforms:
            y = transform.forward(x)
            v = transform.velocity_forward(v, x, y)
        return v

    def velocity_x_to_fourmomenta(self, v_x, x):
        v = v_x.clone()
        x = x.clone()
        for transform in self.transforms[::-1]:
            y = transform.inverse(x)
            v = transform.velocity_forward(v, x, y)
        return v

    def log_prob_fourmomenta_to_x(self, log_prob_fourmomenta, fourmomenta):
        log_prob = log_prob_fourmomenta.clone()
        x = fourmomenta.clone()
        for transform in self.transforms:
            y = transform.forward(x)
            log_prob = log_prob + transform.logdetjac_forward(x, y)
        return log_prob

    def log_prob_x_to_fourmomenta(self, log_prob_x, x):
        log_prob = log_prob_x.clone()
        x = x.clone()
        for transform in self.transforms[::-1]:
            y = transform.inverse(x)
            log_prob = log_prob + transform.logdetjac_inverse(x, y)
        return log_prob


class Fourmomenta(BaseCoordinates):
    # (E, px, py, pz)
    # this class effectively does nothing,
    # because fourmomenta are already the baseline representation
    def __init__(self):
        self.transforms = []


class PPPM2(BaseCoordinates):
    def __init__(self):
        self.transforms = [tr.EPPP_to_PPPM2()]


class PhiCoordinates(BaseCoordinates):
    # abstract class for coordinates with phi in component 1
    def get_trajectory(self, x1, x2, t):
        v_t = x2 - x1
        v_t[..., 1] = ensure_angle(v_t[..., 1])
        x_t = x1 + t * v_t
        x_t[..., 1] = ensure_angle(x_t[..., 1])
        return x_t, v_t


class PtPhiEtaE(PhiCoordinates):
    # (pt, phi, eta, E)
    def __init__(self):
        self.transforms = [tr.EPPP_to_PtPhiEtaE()]


class PtPhiEtaM2(PhiCoordinates):
    def __init__(self):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
        ]


class PPPLogM2(BaseCoordinates):
    # (px, py, pz, E)
    def __init__(self):
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
        ]


class LogPtPhiEtaE(PhiCoordinates):
    # (log(pt), phi, eta, E)
    def __init__(self, pt_min, units):
        self.transforms = [tr.EPPP_to_PtPhiEtaE(), tr.Pt_to_LogPt(pt_min, units)]


class PtPhiEtaLogM2(PhiCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
        ]


class LogPtPhiEtaM2(PhiCoordinates):
    # (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, units):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
        ]


class LogPtPhiEtaLogM2(PhiCoordinates):
    # (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
        ]
