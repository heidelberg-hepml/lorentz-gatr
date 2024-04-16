import torch
import experiments.eventgen.transforms as tr

from experiments.eventgen.helpers import ensure_angle

torch.autograd.set_detect_anomaly(True)

DTYPE = torch.float32


def convert_coordinates(x1, coordinates1, coordinates2):
    if type(coordinates1) == type(coordinates2):
        # no conversion necessary
        x2 = x1
    else:
        # go the long way to fourmomenta and back (could be improved)
        fourmomenta = coordinates1.x_to_fourmomenta(x1)
        x2 = coordinates2.fourmomenta_to_x(fourmomenta)
    return x2


def convert_velocity(v1, x1, coordinates1, coordinates2):
    if type(coordinates1) == type(coordinates2):
        # no conversion necessary
        v2, x2 = v1, x1
    else:
        # go the long way to fourmomenta and back (could be improved)
        v_fourmomenta, fourmomenta = coordinates1.velocity_x_to_fourmomenta(v1, x1)
        v2, x2 = coordinates2.velocity_fourmomenta_to_x(v_fourmomenta, fourmomenta)
    return v2, x2


class BaseCoordinates:
    """
    Class that implements transformations
    from fourmomenta to an abstract set of variables
    Heavily uses functionality from Transforms classes
    """

    def __init__(self):
        self.transforms = []

    def init_fit(self, fourmomenta_list):
        # only does something for FitNormal()
        # requires that FitNormal() comes last in self.transforms
        x_list = [fourmomenta.clone() for fourmomenta in fourmomenta_list]
        for transform in self.transforms[:-1]:
            x_list = [transform.forward(x) for x in x_list]
        self.transforms[-1].init_fit(x_list)

    def init_unit(self, particles_list):
        self.transforms[-1].init_unit(particles_list)

    def get_trajectory(self, x1, x2, t):
        v_t = x2 - x1
        x_t = x1 + t * v_t
        return x_t, v_t

    def fourmomenta_to_x(self, x):
        x = x.to(dtype=DTYPE)
        for transform in self.transforms:
            x = transform.forward(x)
        return x.to(dtype=torch.float32)

    def x_to_fourmomenta(self, x):
        x = x.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            x = transform.inverse(x)
        return x.to(dtype=torch.float32)

    def velocity_fourmomenta_to_x(self, v, x):
        v, x = v.to(dtype=DTYPE), x.to(dtype=DTYPE)
        for transform in self.transforms:
            y = transform.forward(x)
            v = transform.velocity_forward(v, x, y)
            x = y
        return v.to(dtype=torch.float32), x.to(dtype=torch.float32)

    def velocity_x_to_fourmomenta(self, v, x):
        v, x = v.to(dtype=DTYPE), x.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            y = transform.inverse(x)
            v = transform.velocity_inverse(v, x, y)
            x = y
        return v.to(dtype=torch.float32), x.to(dtype=torch.float32)

    def log_prob_fourmomenta_to_x(self, log_prob, x):
        log_prob, x = log_prob.to(dtype=DTYPE), x.to(dtype=DTYPE)
        for transform in self.transforms:
            y = transform.forward(x)
            log_prob = log_prob + transform.logdetjac_forward(x, y)
            x = y
        return log_prob.to(dtype=torch.float32), x.to(dtype=torch.float32)

    def log_prob_x_to_fourmomenta(self, log_prob, x):
        log_prob, x = log_prob.to(dtype=DTYPE), x.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            y = transform.inverse(x)
            log_prob = log_prob + transform.logdetjac_inverse(x, y)
            x = y
        return log_prob.to(dtype=torch.float32), x.to(dtype=torch.float32)


class Fourmomenta(BaseCoordinates):
    # (E, px, py, pz)
    # this class effectively does nothing,
    # because fourmomenta are already the baseline representation
    def __init__(self):
        self.transforms = [tr.EmptyTransform()]


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


class EPhiPtPz(PhiCoordinates):
    # (E, phi, pt, pz)
    def __init__(self):
        self.transforms = [tr.EPPP_to_EPhiPtPz()]


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
    # (px, py, pz, log(m^2))
    def __init__(self):
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
        ]


class FittedPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self):
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
            tr.FitNormal([]),
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


class FittedLogPtPhiEtaLogM2(PhiCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
            tr.FitNormal([1]),
        ]
