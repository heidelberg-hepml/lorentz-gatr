import torch
import experiments.eventgen.transforms as tr

DTYPE = torch.float64


class BaseCoordinates:
    """
    Class that implements transformations
    from fourmomenta to an abstract set of variables
    Heavily uses functionality from Transforms classes
    """

    def __init__(self):
        self.contains_phi = False
        self.contains_mass = False
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
        self.contains_mass = True
        self.transforms = [tr.EPPP_to_PPPM2()]


class EPhiPtPz(BaseCoordinates):
    # (E, phi, pt, pz)
    def __init__(self):
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_EPhiPtPz()]


class PtPhiEtaE(BaseCoordinates):
    # (pt, phi, eta, E)
    def __init__(self):
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE()]


class PtPhiEtaM2(BaseCoordinates):
    def __init__(self):
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
        ]


class PPPLogM2(BaseCoordinates):
    # (px, py, pz, log(m^2))
    def __init__(self):
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
        ]


class StandardPPPLogM2(BaseCoordinates):
    # fitted (px, py, pz, log(m^2))
    def __init__(self, onshell_list=[]):
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PPPM2(),
            tr.M2_to_LogM2(),
            tr.StandardNormal([], onshell_list),
        ]


class LogPtPhiEtaE(BaseCoordinates):
    # (log(pt), phi, eta, E)
    def __init__(self, pt_min, units):
        self.contains_phi = True
        self.transforms = [tr.EPPP_to_PtPhiEtaE(), tr.Pt_to_LogPt(pt_min, units)]


class PtPhiEtaLogM2(BaseCoordinates):
    # (pt, phi, eta, log(m^2))
    def __init__(self):
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.M2_to_LogM2(),
        ]


class LogPtPhiEtaM2(BaseCoordinates):
    # (log(pt), phi, eta, m^2)
    def __init__(self, pt_min, units):
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
        ]


class LogPtPhiEtaLogM2(BaseCoordinates):
    # (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units):
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
        ]


class StandardLogPtPhiEtaLogM2(BaseCoordinates):
    # Fitted (log(pt), phi, eta, log(m^2)
    def __init__(self, pt_min, units, onshell_list=[]):
        self.contains_phi = True
        self.contains_mass = True
        self.transforms = [
            tr.EPPP_to_PtPhiEtaE(),
            tr.PtPhiEtaE_to_PtPhiEtaM2(),
            tr.Pt_to_LogPt(pt_min, units),
            tr.M2_to_LogM2(),
            tr.StandardNormal([1], onshell_list),
        ]
