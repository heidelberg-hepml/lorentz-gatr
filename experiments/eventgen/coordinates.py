import torch
import experiments.eventgen.transforms as tr

from experiments.eventgen.helpers import ensure_angle

DTYPE = torch.float64


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
        # only does something for StandardNormal()
        # requires that StandardNormal() comes last in self.transforms
        x_list = [fourmomenta.clone() for fourmomenta in fourmomenta_list]
        for transform in self.transforms[:-1]:
            x_list = [transform.forward(x) for x in x_list]
        self.transforms[-1].init_fit(x_list)

    def init_unit(self, particles_list):
        self.transforms[-1].init_unit(particles_list)

    def get_metric(self, x1, x2):
        # default: euclidean metric
        se = (x1 - x2) ** 2 / 2
        return se.sum(dim=[-1, -2])

    def get_trajectory(self, x_target, x_base, t, inner_trajectory_func=None):
        # inner_trajectory_func is an ugly construction
        # to give control over which inner trajectory function to use
        # (happy about any suggestions for how to do this better)
        # (only used in MFM right now)
        if inner_trajectory_func is None:
            inner_trajectory_func = self._get_trajectory
        return inner_trajectory_func(x_target, x_base, t)

    def _get_trajectory(self, x_target, x_base, t):
        # default: straight trajectories
        v_t = x_base - x_target
        x_t = x_target + t * v_t
        return x_t, v_t

    def fourmomenta_to_x(self, a_in):
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        for transform in self.transforms:
            a = transform.forward(a)
        return a.to(dtype=a_in.dtype)

    def x_to_fourmomenta(self, a_in):
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            a = transform.inverse(a)
        return a.to(dtype=a_in.dtype)

    def velocity_fourmomenta_to_x(self, v_in, a_in):
        assert torch.isfinite(a_in).all() and torch.isfinite(v_in).all()
        v, a = v_in.to(dtype=DTYPE), a_in.to(dtype=DTYPE)
        for transform in self.transforms:
            b = transform.forward(a)
            v = transform.velocity_forward(v, a, b)
            a = b
        return v.to(dtype=v_in.dtype), a.to(dtype=a_in.dtype)

    def velocity_x_to_fourmomenta(self, v_in, a_in):
        assert torch.isfinite(a_in).all() and torch.isfinite(v_in).all()
        v, a = v_in.to(dtype=DTYPE), a_in.to(dtype=DTYPE)
        for transform in self.transforms[::-1]:
            b = transform.inverse(a)
            v = transform.velocity_inverse(v, a, b)
            a = b
        return v.to(dtype=v_in.dtype), a.to(dtype=a_in.dtype)

    def logdetjac_fourmomenta_to_x(self, a_in):
        # logdetjac = log|da/db| = -log|db/da| with a=fourmomenta, b=x
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        b = self.transforms[0].forward(a)
        logdetjac = -self.transforms[0].logdetjac_forward(a, b)
        a = b
        for transform in self.transforms[1:]:
            b = transform.forward(a)
            logdetjac -= transform.logdetjac_forward(a, b)
            a = b
        return logdetjac.to(dtype=a_in.dtype), a.to(dtype=a_in.dtype)

    def logdetjac_x_to_fourmomenta(self, a_in):
        # logdetjac = log|da/db| = -log|db/da| with a=x, b=fourmomenta
        assert torch.isfinite(a_in).all()
        a = a_in.to(dtype=DTYPE)
        b = self.transforms[-1].inverse(a)
        logdetjac = -self.transforms[-1].logdetjac_inverse(a, b)
        a = b
        for transform in self.transforms[::-1][1:]:
            b = transform.inverse(a)
            logdetjac -= transform.logdetjac_inverse(a, b)
            a = b
        return logdetjac.to(dtype=a_in.dtype), a.to(dtype=a_in.dtype)


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
    def get_trajectory(self, x_target, x_base, t, inner_trajectory_func=None):
        x_t, v_t = super().get_trajectory(x_target, x_base, t, inner_trajectory_func)
        x_t[..., 1] = ensure_angle(x_t[..., 1])
        v_t[..., 1] = ensure_angle(v_t[..., 1])
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


class StandardPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self, onshell_list=[]):
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal([], onshell_list),
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


class StandardLogPtPhiEtaLogM2(PhiCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units, onshell_list=[]):
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
            tr.StandardNormal([1], onshell_list),
        ]
