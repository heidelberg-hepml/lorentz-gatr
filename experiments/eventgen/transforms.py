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

    def forward(self, x):
        y = self._forward(x)
        assert torch.isfinite(y).all()
        return y

    def inverse(self, x):
        y = self._inverse(x)
        assert torch.isfinite(y).all()
        return y

    def velocity_forward(self, v_x, x, y):
        # v_y = dy/dx * v_x
        jac = self._jac_forward(x, y)
        v_y = torch.einsum("...ij,...j->...i", jac, v_x)
        assert torch.isfinite(v_y).all()
        return v_y

    def velocity_inverse(self, v_y, y, x):
        # v_x = dx/dy * v_y
        jac = self._jac_inverse(y, x)
        v_x = torch.einsum("...ij,...j->...i", jac, v_y)
        assert torch.isfinite(v_x).all()
        return v_x

    def logdetjac_forward(self, x, y):
        logdetjac = self._detjac_forward(x, y).log()
        assert torch.isfinite(logdetjac)
        return logdetjac

    def logdetjac_inverse(self, y, x):
        # log(det(J^-1)) = log(1/det(J)) = -log(det(J))
        logdetjac = -self._detjac_forward(x, y).log()
        assert torch.isfinite(logdetjac)
        return logdetjac

    def _forward(self, x):
        raise NotImplementedError

    def _inverse(self, x):
        raise NotImplementedError

    def _jac_forward(self, x, y):
        raise NotImplementedError

    def _jac_inverse(self, y, x):
        raise NotImplementedError

    def _detjac_forward(self, x, y):
        raise NotImplementedError


class EPPP_to_PPPM2(BaseTransform):
    def _forward(self, eppp):
        E, px, py, pz = unpack_last(eppp)

        m2 = E**2 - (px**2 + py**2 + pz**2)
        m2 = stay_positive(m2)
        return torch.stack((px, py, pz, m2), dim=-1)

    def _inverse(self, pppm2):
        px, py, pz, m2 = unpack_last(pppm2)
        m2 = stay_positive(m2)

        E = torch.sqrt(m2 + (px**2 + py**2 + pz**2))
        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, pppm2):
        E, px, py, pz = unpack_last(eppp)
        px, py, pz, m2 = unpack_last(pppm2)

        # jac_ij = dpppm2_i / deppp_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_E = torch.stack((zero, zero, zero, 2 * E), dim=-1)
        jac_px = torch.stack((one, zero, zero, -2 * px), dim=-1)
        jac_py = torch.stack((zero, one, zero, -2 * py), dim=-1)
        jac_pz = torch.stack((zero, zero, one, -2 * pz), dim=-1)
        return torch.stack((jac_E, jac_px, jac_py, jac_pz), dim=-1)

    def _jac_inverse(self, pppm2, eppp):
        E, px, py, pz = unpack_last(eppp)
        px, py, pz, m2 = unpack_last(pppm2)

        # jac_ij = deppp_i / dpppm2_j
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_px = torch.stack((px / E, one, zero, zero), dim=-1)
        jac_py = torch.stack((py / E, zero, one, zero), dim=-1)
        jac_pz = torch.stack((pz / E, zero, zero, one), dim=-1)
        jac_m2 = torch.stack((1 / (2 * E), zero, zero, zero), dim=-1)
        return torch.stack((jac_px, jac_py, jac_pz, jac_m2), dim=-1)

    def _detjac_forward(self, eppp, pppm2):
        E, px, py, pz = unpack_last(eppp)
        px, py, pz, m2 = unpack_last(pppm2)
        return 2 * E


class EPPP_to_PtPhiEtaE(BaseTransform):
    def _forward(self, eppp):
        E, px, py, pz = unpack_last(eppp)

        pt = torch.sqrt(px**2 + py**2)
        phi = torch.arctan2(py, px)
        p_abs = torch.sqrt(pt**2 + pz**2)
        eta = stable_arctanh(pz / p_abs)  # torch.arctanh(pz / p_abs)
        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        assert torch.isfinite(eta).all()

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _inverse(self, ptphietae):
        pt, phi, eta, E = unpack_last(ptphietae)

        eta = eta.clamp(min=-CUTOFF, max=CUTOFF)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)

        return torch.stack((E, px, py, pz), dim=-1)

    def _jac_forward(self, eppp, ptphietae):
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

    def _jac_inverse(self, ptphietae, eppp):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # jac_ij = dfourmomenta_i / djetmomenta_j
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

    def _detjac_forward(self, eppp, ptphietae):
        E, px, py, pz = unpack_last(eppp)
        pt, phi, eta, E = unpack_last(ptphietae)

        # det (dptphietae / dfourmomenta)
        p_abs = pt * torch.cosh(eta)
        return 1 / p_abs


class PtPhiEtaE_to_PtPhiEtaM2(BaseTransform):
    def _forward(self, ptphietae):
        pt, phi, eta, E = unpack_last(ptphietae)

        p_abs = pt * torch.cosh(eta)
        m2 = E**2 - p_abs**2
        m2 = stay_positive(m2)
        return torch.stack((pt, phi, eta, m2), dim=-1)

    def _inverse(self, ptphietam2):
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        m2 = stay_positive(m2)
        p_abs = pt * torch.cosh(eta)
        E = torch.sqrt(m2 + p_abs**2)

        return torch.stack((pt, phi, eta, E), dim=-1)

    def _jac_forward(self, ptphietae, ptphietam2):
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

    def _jac_inverse(self, ptphietam2, ptphietae):
        pt, phi, eta, E = unpack_last(ptphietae)
        pt, phi, eta, m2 = unpack_last(ptphietam2)

        # jac_ij = dptphietae_i / dptphietam2_j
        p_abs = pt * torch.cosh(eta)
        zero, one = torch.zeros_like(E), torch.ones_like(E)
        jac_pt = torch.stack((one, zero, zero, pt * torch.cosh(eta) ** 2 / E), dim=-1)
        jac_phi = torch.stack((zero, one, zero, zero), dim=-1)
        jac_eta = torch.stack(
            (zero, zero, one, pt**2 * torch.sinh(2 * eta) / (2 * E)), dim=-1
        )
        jac_m2 = torch.stack((zero, zero, zero, 1 / (2 * E)), dim=-1)

        return torch.stack((jac_pt, jac_phi, jac_eta, jac_m2), dim=-1)


class M2_to_LogM2(BaseTransform):
    def _forward(self, xm2):
        x1, x2, x3, m2 = unpack_last(xm2)
        m2 = m2.clamp(min=EPS2)
        logm2 = torch.log(m2 + EPS1)
        return torch.stack((x1, x2, x3, logm2), dim=-1)

    def _inverse(self, xlogm2):
        x1, x2, x3, logm2 = unpack_last(xlogm2)
        m2 = logm2.clamp(max=CUTOFF).exp() - EPS1
        return torch.stack((x1, x2, x3, m2), dim=-1)

    def _jac_forward(self, xm2, logxm2):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxlogm2_i / dxm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, 1 / (m2 + EPS1)), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_m2), dim=-1)

    def _jac_inverse(self, logxm2, xm2):
        x1, x2, x3, m2 = unpack_last(xm2)

        # jac_ij = dxm2_i / dxlogm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_logm2 = torch.stack((zero, zero, zero, m2 + EPS1), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_logm2), dim=-1)

    def _detjac_forward(self, xm2, logxm2):
        x1, x2, x3, m2 = unpack_last(xm2)
        return 1 / (m2 + EPS1)


class Pt_to_LogPt(BaseTransform):
    def __init__(self, pt_min, units):
        self.pt_min = torch.tensor(pt_min) / units

    def get_dpt(self, pt):
        return torch.clamp(pt - self.pt_min[: pt.shape[-1]].to(pt.device), min=EPS2)

    def _forward(self, ptx):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        logpt = torch.log(dpt + EPS1)
        return torch.stack((logpt, x1, x2, x3), dim=-1)

    def _inverse(self, logptx):
        logpt, x1, x2, x3 = unpack_last(logptx)
        pt = (
            logpt.clamp(max=CUTOFF).exp()
            + self.pt_min[: logpt.shape[-1]].to(logpt.device)
            - EPS1
        )
        return torch.stack((pt, x1, x2, x3), dim=-1)

    def _jac_forward(self, ptx, logptx):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dlogptx_i / dptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_pt = torch.stack(
            (
                1 / (dpt + EPS1),
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

    def _jac_inverse(self, logptx, ptx):
        pt, x1, x2, x3 = unpack_last(ptx)

        # jac_ij = dptx_i / dlogptx_j
        zero, one = torch.zeros_like(pt), torch.ones_like(pt)
        dpt = self.get_dpt(pt)
        jac_logpt = torch.stack((dpt + EPS1, zero, zero, zero), dim=-1)
        jac_x1 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, zero, one), dim=-1)
        return torch.stack((jac_logpt, jac_x1, jac_x2, jac_x3), dim=-1)

    def _detjac_forward(self, ptx, logptx):
        pt, x1, x2, x3 = unpack_last(ptx)
        dpt = self.get_dpt(pt)
        return 1 / (dpt + EPS2)


class M2rescale(BaseTransform):
    def __init__(self, mass_scale):
        self.m2_scale = mass_scale**2

    def _forward(self, xm2):
        x1, x2, x3, m2 = unpack_last(xm2)
        m2_mod = m2 / self.m2_scale
        return torch.stack((x1, x2, x3, m2_mod), dim=-1)

    def _inverse(self, xm2_mod):
        x1, x2, x3, m2_mod = unpack_last(xm2_mod)
        m2 = m2_mod * self.m2_scale
        return torch.stack((x1, x2, x3, m2), dim=-1)

    def _jac_forward(self, xm2, xm2_mod):
        _, _, _, m2 = unpack_last(xm2)

        # jac_ij = xm2_mod_i / dxm2_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2 = torch.stack((zero, zero, zero, 1 / self.m2_scale * one), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_m2), dim=-1)

    def _jac_inverse(self, xm2_mod, xm2):
        _, _, _, m2 = unpack_last(xm2)

        # jac_ij = dxm2_i / dxm2_mod_j
        zero, one = torch.zeros_like(m2), torch.ones_like(m2)
        jac_x1 = torch.stack((one, zero, zero, zero), dim=-1)
        jac_x2 = torch.stack((zero, one, zero, zero), dim=-1)
        jac_x3 = torch.stack((zero, zero, one, zero), dim=-1)
        jac_m2_mod = torch.stack((zero, zero, zero, self.m2_scale * one), dim=-1)
        return torch.stack((jac_x1, jac_x2, jac_x3, jac_m2_mod), dim=-1)

    def _detjac_forward(self, xm2, xm2_mod):
        return 1 / self.m2_scale
