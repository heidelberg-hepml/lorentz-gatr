import torch
from torch import nn

from experiments.eventgen.helpers import (
    unpack_last,
    EPS1,
    EPS2,
    CUTOFF,
    stable_arctanh,
    stay_positive,
)


class BaseTransform(nn.Module):
    """
    Abstract class for transformations between two coordinate systems
    For CFM, we need forward and inverse transformations,
    the corresponding jacobians for the RCFM aspect
    and log(det(jacobian)) when we want to extract probabilities from the CFM
    """

    def forward(self, x, **kwargs):
        y = self._forward(x, **kwargs)
        assert torch.isfinite(y).all(), self.__class__.__name__
        return y

    def inverse(self, x, **kwargs):
        y = self._inverse(x, **kwargs)
        assert torch.isfinite(y).all(), self.__class__.__name__
        return y

    def velocity_forward(self, v_x, x, y, **kwargs):
        # v_y = dy/dx * v_x
        jac = self._jac_forward(x, y, **kwargs)
        v_y = torch.einsum("...ij,...j->...i", jac, v_x)
        assert torch.isfinite(v_y).all(), self.__class__.__name__
        return v_y

    def velocity_inverse(self, v_y, y, x, **kwargs):
        # v_x = dx/dy * v_y
        jac = self._jac_inverse(y, x, **kwargs)
        v_x = torch.einsum("...ij,...j->...i", jac, v_y)
        assert torch.isfinite(v_x).all(), self.__class__.__name__
        return v_x

    def logdetjac_forward(self, x, y, **kwargs):
        # log(det(J))
        # J = dy/dx
        logdetjac = torch.log(self._detjac_forward(x, y, **kwargs).abs() + EPS2).sum(
            dim=-1, keepdims=True
        )
        assert torch.isfinite(logdetjac).all(), self.__class__.__name__
        return logdetjac

    def logdetjac_inverse(self, y, x, **kwargs):
        # log(det(J^-1)) = log(1/det(J)) = -log(det(J))
        # J = dy/dx
        logdetjac = -torch.log(self._detjac_forward(x, y, **kwargs).abs() + EPS2).sum(
            dim=-1, keepdims=True
        )
        assert torch.isfinite(logdetjac).all(), self.__class__.__name__
        return logdetjac

    def _forward(self, x, **kwargs):
        raise NotImplementedError

    def _inverse(self, x, **kwargs):
        raise NotImplementedError

    def _jac_forward(self, x, y, **kwargs):
        raise NotImplementedError

    def _jac_inverse(self, y, x, **kwargs):
        raise NotImplementedError

    def _detjac_forward(self, x, y, **kwargs):
        raise NotImplementedError

    def init_fit(self, xs):
        # currently only needed for StandardNormal()
        # default: do nothing
        pass

    def init_unit(self, xs):
        # for debugging and tests
        pass


class EmptyTransform(BaseTransform):
    # empty transform
    # needed for formal reasons
    def forward(self, x, **kwargs):
        return x

    def inverse(self, x, **kwargs):
        return x

    def velocity_forward(self, v, x, y, **kwargs):
        return v

    def velocity_inverse(self, v, y, x, **kwargs):
        return v

    def logdetjac_forward(self, x, y, **kwargs):
        return torch.zeros_like(x[..., 0])

    def logdetjac_inverse(self, x, y, **kwargs):
        return torch.zeros_like(x[..., 0])


class EPPP_to_PPPM2(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        m2 = E**2 - (px**2 + py**2 + pz**2)
        m2 = stay_positive(m2)
        return torch.stack((px, py, pz, m2), dim=-1)

    def _inverse(self, pppm2, **kwargs):
        px, py, pz, m2 = unpack_last(pppm2)
        m2 = stay_positive(m2)

        E = torch.sqrt(m2 + (px**2 + py**2 + pz**2))
        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, pppm2, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        # jac_ij = dpppm2_i / deppp_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((zero, zero, zero, 2 * E), dim=-1)
        jac_px = torch.stack((one, zero, zero, -2 * px), dim=-1)
        jac_py = torch.stack((zero, one, zero, -2 * py), dim=-1)
        jac_pz = torch.stack((zero, zero, one, -2 * pz), dim=-1)
        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, pppm2, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        # jac_ij = deppp_i / dpppm2_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_px = torch.stack((px / E, one, zero, zero), dim=-1)
        jac_py = torch.stack((py / E, zero, one, zero), dim=-1)
        jac_pz = torch.stack((pz / E, zero, zero, one), dim=-1)
        jac_m2 = torch.stack((1 / (2 * E), zero, zero, zero), dim=-1)
        return torch.stack((jac_px, jac_py, jac_pz, jac_m2), dim=-1)

    def _detjac_forward(self, eppp, pppm2, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        return 2 * E


class EPPP_to_EPhiPtPz(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        pt = torch.sqrt(px**2 + py**2)
        phi = torch.arctan2(py, px)
        return torch.stack((E, phi, pt, pz), dim=-1)

    def _inverse(self, ephiptpz, **kwargs):
        E, phi, pt, pz = unpack_last(ephiptpz)

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, ephiptpz, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        E, phi, pt, pz = unpack_last(ephiptpz)

        # jac_ij = dephiptpz_i / dfourmomenta_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)
        jac_px = torch.stack(
            (zero, -py / pt**2, px / pt, zero),
            dim=-1,
        )
        jac_py = torch.stack(
            (zero, px / pt**2, py / pt, zero),
            dim=-1,
        )
        jac_pz = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, ephiptpz, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        E, phi, pt, pz = unpack_last(ephiptpz)

        # jac_ij = dfourmomenta_i / dephiptpz_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)
        jac_phi = torch.stack(
            (zero, -pt * torch.sin(phi), pt * torch.cos(phi), zero), dim=-1
        )
        jac_pt = torch.stack((zero, torch.cos(phi), torch.sin(phi), zero), dim=-1)
        jac_pz = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_E, jac_phi, jac_pt, jac_pz), dim=-1)

    def _detjac_forward(self, eppp, ephiptpz, **kwargs):
        E, phi, pt, pz = unpack_last(ephiptpz)

        # det (dephiptpz / dfourmomenta)
        return 1 / pt


class EPPP_to_PtPhiEtaE(BaseTransform):
    def _forward(self, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)

        pt = torch.sqrt(px**2 + py**2)
        phi = torch.arctan2(py, px)
        p_abs = torch.sqrt(pt**2 + pz**2)
        eta = stable_arctanh(pz / p_abs)  # torch.arctanh(pz / p_abs)
        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        assert torch.isfinite(eta).all()

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _inverse(self, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)

        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, ptphietae, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # jac_ij = dptphietae_i / dfourmomenta_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((zero, zero, zero, one), dim=-1)
        jac_px = torch.stack(
            (px / pt, -py / pt**2, -px * pz / (pt**3 * torch.cosh(eta)), zero),
            dim=-1,
        )
        jac_py = torch.stack(
            (py / pt, px / pt**2, -py * pz / (pt**3 * torch.cosh(eta)), zero),
            dim=-1,
        )
        jac_pz = torch.stack((zero, zero, 1 / (pt * torch.cosh(eta)), zero), dim=-1)

        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, ptphietae, eppp, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # jac_ij = dfourmomenta_i / dptphietae_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack(
            (zero, torch.cos(phi), torch.sin(phi), torch.sinh(eta)), dim=-1
        )
        jac_phi = torch.stack(
            (zero, -pt * torch.sin(phi), pt * torch.cos(phi), zero), dim=-1
        )
        jac_eta = torch.stack((zero, zero, zero, pt * torch.cosh(eta)), dim=-1)
        jac_E = torch.stack((one, zero, zero, zero), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_E), dim=-1)

    def _detjac_forward(self, eppp, ptphietae, **kwargs):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # det (dptphietae / dfourmomenta)
        return 1 / (pt**2 * torch.cosh(eta))


class PtPhiEtaE_to_PtPhiEtaM2(BaseTransform):
    def _forward(self, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)

        p_abs = pt * torch.cosh(eta)
        m2 = E**2 - p_abs**2
        m2 = stay_positive(m2)
        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _inverse(self, ptphietam2, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        m2 = stay_positive(m2)
        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        p_abs = pt * torch.cosh(eta)
        E = torch.sqrt(m2 + p_abs**2)

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _jac_forward(self, ptphietae, ptphietam2, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        # jac_ij = dptphietam2_i / dptphietae_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack((one, zero, zero, -2 * pt * torch.cosh(eta) ** 2), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack(
            (zero, zero, one, -(pt**2) * torch.sinh(2 * eta)), dim=-1
        )
        jac_E = torch.stack((zero, zero, zero, 2 * E), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_E), dim=-1)

    def _jac_inverse(self, ptphietam2, ptphietae, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        # jac_ij = dptphietae_i / dptphietam2_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack((one, zero, zero, pt * torch.cosh(eta) ** 2 / E), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack(
            (zero, zero, one, pt**2 * torch.sinh(2 * eta) / (2 * E)), dim=-1
        )
        jac_m2 = torch.stack((zero, zero, zero, 1 / (2 * E)), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _detjac_forward(self, ptphietae, ptphietam2, **kwargs):
        pt, phi, eta, E = unpack_last(ptphietae)
        return 2 * E


class M2_to_LogM2(BaseTransform):
    def _forward(self, xm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)
        m2 = m2.clamp(min=EPS2)
        logm2 = torch.log(m2 + EPS1)
        return torch.stack((x1, x2, x3, logm2), dim=-1)

    def _inverse(self, xlogm2, **kwargs):
        x1, x2, x3, logm2 = unpack_last(xlogm2)
        m2 = logm2.clamp(max=CUTOFF).exp() - EPS1
        return torch.stack((x1, x2, x3, m2), dim=-1)

    def _jac_forward(self, xm2, logxm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxlogm2_i / dxm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, 1 / (m2 + EPS1 + EPS2)), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_m2), dim=-1)

    def _jac_inverse(self, logxm2, xm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxm2_i / dxlogm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_logm2 = torch.stack((zero, zero, zero, m2 + EPS1), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_logm2), dim=-1)

    def _detjac_forward(self, xm2, logxm2, **kwargs):
        x1, x2, x3, m2 = unpack_last(xm2)
        return 1 / (m2 + EPS1 + EPS2)


class Pt_to_LogPt(BaseTransform):
    def __init__(self, pt_min, units):
        self.pt_min = (
            pt_min if isinstance(pt_min, torch.Tensor) else torch.tensor(pt_min)
        )
        self.pt_min /= units

    def get_dpt(self, pt, **kwargs):
        return torch.clamp(pt - self.pt_min[: pt.shape[-1]].to(pt.device), min=EPS2)

    def _forward(self, ptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        logpt = torch.log(dpt + EPS1)
        return torch.stack((logpt, x1, x2, x3), dim=-1)

    def _inverse(self, logptx, **kwargs):
        logpt, x1, x2, x3 = unpack_last(logptx)
        pt = (
            logpt.clamp(max=CUTOFF).exp()
            + self.pt_min[: logpt.shape[-1]].to(logpt.device)
            - EPS1
        )
        return torch.stack((pt, x1, x2, x3), dim=-1)

    def _jac_forward(self, ptx, logptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dlogptx_i / dptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_pt = torch.stack(
            (
                1 / (dpt + EPS1 + EPS2),
                zero,
                zero,
                zero,
            ),
            dim=-1,
        )
        jac_x1 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, zero, one), dim=-1)
        return torch.stack((jac_pt, jac_x1, jac_x2, jac_x3), dim=-1)

    def _jac_inverse(self, logptx, ptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dptx_i / dlogptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_logpt = torch.stack((dpt + EPS1, zero, zero, zero), dim=-1)
        jac_x1 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, zero, one), dim=-1)
        return torch.stack((jac_logpt, jac_x1, jac_x2, jac_x3), dim=-1)

    def _detjac_forward(self, ptx, logptx, **kwargs):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        return 1 / (dpt + EPS1 + EPS2)


class StandardNormal(BaseTransform):
    # standardize to unit normal distribution
    # particle- and process-wise mean and std are determined by initial_fit
    # note: this transform will always come last in the self.transforms list of a coordinates class
    def __init__(self, dims_fixed=[], onshell_list=None):
        self.dims_fixed = dims_fixed
        self.onshell_list = onshell_list

    def init_fit(self, xs):
        n_particles = [x.shape[-2] for x in xs]
        assert len(n_particles) == len(
            set(n_particles)
        ), f"n_particles should have unique elements, but n_particles={n_particles}"
        self.params = {n_p: {"mean": None, "std": None} for n_p in n_particles}
        for i, n_p in enumerate(n_particles):
            assert len(xs[i].shape) == 3
            self.params[n_p]["mean"] = xs[i].mean(dim=0)
            std = xs[i].std(dim=0)
            if self.onshell_list is not None:
                std[self.onshell_list, 3] = 1.0
            assert torch.isfinite(std).all()
            self.params[n_p]["std"] = std

            # do not fit some distributions
            self.params[n_p]["mean"][..., self.dims_fixed] = 0.0
            self.params[n_p]["std"][..., self.dims_fixed] = 1.0

    def init_unit(self, n_particles):
        # initialize to zero mean and unit std
        # only for debugging and tests
        self.params = {n_p: {"mean": None, "std": None} for n_p in n_particles}
        for i, n_p in enumerate(n_particles):
            self.params[n_p]["mean"] = torch.zeros(n_p, 4)
            self.params[n_p]["std"] = torch.ones(n_p, 4)

    def get_mean_std(self, x):
        params = self.params[x.shape[-2]]
        return params["mean"].to(x.device, dtype=x.dtype), params["std"].to(
            x.device, dtype=x.dtype
        )

    def _forward(self, x, **kwargs):
        mean, std = self.get_mean_std(x)
        xunit = (x - mean) / std
        return xunit

    def _inverse(self, xunit, **kwargs):
        mean, std = self.get_mean_std(xunit)
        x = xunit * std + mean
        return x

    def _jac_forward(self, x, xunit, **kwargs):
        std = self.get_mean_std(x)[1]
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        jac[..., torch.arange(4), torch.arange(4)] = 1 / std.unsqueeze(0)
        return jac

    def _jac_inverse(self, xunit, x, **kwargs):
        std = self.get_mean_std(x)[1]
        jac = torch.zeros(*x.shape, 4, device=x.device, dtype=x.dtype)
        jac[..., torch.arange(4), torch.arange(4)] = std.unsqueeze(0)
        return jac

    def _detjac_forward(self, x, xunit, **kwargs):
        std = self.get_mean_std(x)[1]
        detjac = 1 / torch.prod(std, dim=-1)
        detjac = detjac.unsqueeze(0).expand(x.shape[0], x.shape[1])
        return detjac


class PtPhiEtaM2_to_PtPhiEtaM2Relative(BaseTransform):
    # takes 'jet' (in PtPhiEtaM2 coordinates) as extra keyword argument

    def _forward(self, ptphietam2, jet, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)
        assert jet.shape == ptphietam2.shape
        pt_jet, phi_jet, eta_jet, m2_jet = unpack_last(jet)

        pt_rel = pt / pt_jet
        phi_rel = phi - phi_jet
        eta_rel = eta - eta_jet

        return torch.stack((pt_rel, phi_rel, eta_rel, m2), dim=-1)

    def _inverse(self, ptphietam2_rel, jet, **kwargs):
        pt_rel, phi_rel, eta_rel, m2 = unpack_last(ptphietam2_rel)
        assert jet.shape == ptphietam2_rel.shape
        pt_jet, phi_jet, eta_jet, m2_jet = unpack_last(jet)

        pt = pt_rel * pt_jet
        phi = phi_rel + phi_jet
        eta = eta_rel + eta_jet

        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _jac_forward(self, ptphietam2, ptphietam2_rel, jet, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)
        assert jet.shape == ptphietam2.shape
        pt_jet, phi_jet, eta_jet, m2_jet = unpack_last(jet)

        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        jac_pt = torch.stack((1 / pt_jet, zero, zero, zero), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)

    def _jac_inverse(self, ptphietam2_rel, ptphietam2, jet, **kwargs):
        pt, phi, eta, m2 = unpack_last(ptphietam2)
        assert jet.shape == ptphietam2.shape
        pt_jet, phi_jet, eta_jet, m2_jet = unpack_last(jet)

        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        jac_pt_rel = torch.stack((pt_jet, zero, zero, zero), dim=-1)
        jac_phi_rel = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta_rel = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt_rel, jac_phi_rel, jac_eta_rel, jac_m2), dim=-1)

    def _detjac_forward(self, ptphietam2, ptphietam2_rel, jet, **kwargs):
        assert jet.shape == ptphietam2.shape
        pt_jet, phi_jet, eta_jet, m2_jet = unpack_last(jet)

        return 1 / pt_jet


class PtRel_to_LogitPtRel(BaseTransform):
    # note: the logit transform only works if all jet constituents have pt < pt_jet,
    # otherwise pt/pt_jet > 1 and the logit can not be used (can use 'log' then)

    def _forward(self, ptphietam2_rel, **kwargs):
        pt_rel, a, b, c = unpack_last(ptphietam2_rel)
        assert (pt_rel >= 0).all() and (pt_rel <= 1).all()

        # logit transformation
        pt_logit = torch.logit(pt_rel)
        return torch.stack((pt_logit, a, b, c), dim=-1)

    def _inverse(self, logitptphietam2_rel, **kwargs):
        pt_logit, a, b, c = unpack_last(logitptphietam2_rel)

        # inverse logit transformation
        pt_rel = torch.sigmoid(pt_logit)
        return torch.stack((pt_rel, a, b, c), dim=-1)

    def _jac_forward(self, pt_rel, logitpt_rel, **kwargs):
        pt_rel, a, b, c = unpack_last(pt_rel)
        assert (pt_rel >= 0).all() and (pt_rel <= 1).all()

        # jacobian of logit transformation
        zero, one = torch.zeros_like(pt_rel), torch.ones_like(pt_rel)
        jac_pt_rel = torch.stack(
            (1 / (pt_rel * (1 - pt_rel)), zero, zero, zero), dim=-1
        )
        jac_a = torch.stack((zero, one, zero, zero), dim=-1)
        jac_b = torch.stack((zero, zero, one, zero), dim=-1)
        jac_c = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_pt_rel, jac_a, jac_b, jac_c), dim=-1)

    def _jac_inverse(self, pt_rel, logitpt_rel, **kwargs):
        logitpt_rel, a, b, c = unpack_last(pt_rel)

        # jacobian of logit transformation
        zero, one = torch.zeros_like(logitpt_rel), torch.ones_like(logitpt_rel)
        jac_logitpt_rel = torch.stack(
            (
                torch.sigmoid(logitpt_rel) * (1 - torch.sigmoid(logitpt_rel)),
                zero,
                zero,
                zero,
            ),
            dim=-1,
        )
        jac_a = torch.stack((zero, one, zero, zero), dim=-1)
        jac_b = torch.stack((zero, zero, one, zero), dim=-1)
        jac_c = torch.stack((zero, zero, zero, one), dim=-1)

        return torch.stack((jac_logitpt_rel, jac_a, jac_b, jac_c), dim=-1)

    def _detjac_forward(self, pt_rel, logitpt_rel, **kwargs):
        pt_rel, a, b, c = unpack_last(pt_rel)
        assert (pt_rel >= 0).all() and (pt_rel <= 1).all()
        return 1 / (pt_rel * (1 - pt_rel))
