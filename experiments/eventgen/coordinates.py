import torch
from experiments.eventgen.transforms import (
    EPS1,
    CUTOFF,
    ensure_angle,
    get_mass,
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
)


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


class BaseCoordinates:
    # abstract base class for all coordinates
    def __init__(self):
        pass

    def fourmomenta_to_x(self, fourmomenta):
        raise NotImplementedError

    def x_to_fourmomenta(self, x):
        raise NotImplementedError

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, x):
        raise NotImplementedError

    def velocities_x_to_fourmomenta(self, v_x, x, fourmomenta):
        raise NotImplementedError

    def log_prob_x_to_fourmomenta(self, log_prob_x, x, fourmomenta):
        raise NotImplementedError

    def final_checks(self, x):
        # default: do nothing
        return x


class Fourmomenta(BaseCoordinates):
    def fourmomenta_to_x(self, fourmomenta):
        return fourmomenta

    def x_to_fourmomenta(self, x):
        return x

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, x):
        return v_fourmomenta

    def velocities_x_to_fourmomenta(self, v_x, x, fourmomenta):
        return v_x

    def log_prob_x_to_fourmomenta(self, log_prob_x, x, fourmomenta):
        return log_prob_x

    def final_checks(self, fourmomenta):
        # set m[m<0] = -m[m<0] (helps for GATr)
        # should try with and without this setting (can do both for trained model)
        p_abs = (fourmomenta[..., 1:] ** 2).sum(dim=-1)
        mass2 = fourmomenta[..., 0] ** 2 - p_abs
        mass2[mass2 < 0] = -mass2[mass2 < 0]
        fourmomenta[..., 0] = torch.sqrt(mass2 + p_abs)
        return fourmomenta


class PtPhiEtaE(BaseCoordinates):
    def fourmomenta_to_x(self, fourmomenta):
        E, _, _, _ = unpack_last(fourmomenta)

        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        pt, phi, eta, _ = unpack_last(jetmomenta)

        ptphietae = torch.stack([pt, phi, eta, E], dim=-1)
        assert torch.isfinite(ptphietae).all()
        return ptphietae

    def x_to_fourmomenta(self, ptphietae):
        pt, phi, eta, E = unpack_last(ptphietae)

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF))

        fourmomenta = torch.stack((E, px, py, pz), dim=-1)
        assert torch.isfinite(fourmomenta).all()
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, ptphietae):
        pt, phi, eta, E = unpack_last(ptphietae)
        E, px, py, pz = unpack_last(fourmomenta)
        v_E, v_px, v_py, v_pz = unpack_last(v_fourmomenta)

        v_pt = (px * v_px + py * v_py) / pt
        v_phi = (px * v_py - py * v_px) / pt**2
        v_eta = (
            (pt**2 * v_pz - pz * (px * v_px + py * v_py))
            / pt**2
            / torch.sqrt(px**2 + py**2 + pz**2)
        )

        v_ptphietae = torch.stack((v_pt, v_phi, v_eta, v_E), dim=-1)
        assert torch.isfinite(v_ptphietae).all()
        return v_ptphietae

    def velocities_x_to_fourmomenta(self, v_ptphietae, ptphietae, fourmomenta):
        pt, phi, eta, E = unpack_last(ptphietae)
        E, px, py, pz = unpack_last(fourmomenta)
        v_pt, v_phi, v_eta, v_E = unpack_last(v_ptphietae)

        v_px = torch.cos(phi) * v_pt - torch.sin(phi) * pt * v_phi
        v_py = torch.sin(phi) * v_pt + torch.cos(phi) * pt * v_phi
        v_pz = (
            torch.cosh(eta.clamp(min=-CUTOFF, max=CUTOFF)) * pt * v_eta
            + torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF)) * v_pt
        )

        v_fourmomenta = torch.stack((v_E, v_px, v_py, v_pz), dim=-1)
        assert torch.isfinite(v_fourmomenta).all()
        return v_fourmomenta

    def final_checks(self, ptphietae):
        ptphietae[..., 1] = ensure_angle(ptphietae[..., 1])
        return ptphietae


class PPPM(BaseCoordinates):
    def __init__(self, mass_scale=1.0):
        # not properly used yet
        self.mass_scale = mass_scale

    def fourmomenta_to_x(self, fourmomenta):
        _, px, py, pz = unpack_last(fourmomenta)
        mass = get_mass(fourmomenta)
        mass /= self.mass_scale

        pppm = torch.stack((px, py, pz, mass), dim=-1)
        assert torch.isfinite(pppm).all()
        return pppm

    def x_to_fourmomenta(self, pppm):
        px, py, pz, mass = unpack_last(pppm)

        mass *= self.mass_scale
        E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

        fourmomenta = torch.stack((E, px, py, pz), dim=-1)
        assert torch.isfinite(fourmomenta).all()
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, pppm):
        px, py, pz, mass = unpack_last(pppm)
        E, px, py, pz = unpack_last(fourmomenta)
        v_E, v_px, v_py, v_pz = unpack_last(v_fourmomenta)

        v_mass = (E * v_E - px * v_px - py * v_py - pz * v_pz) / mass
        v_mass /= self.mass_scale

        v_pppm = torch.stack((v_px, v_py, v_pz, v_mass), dim=-1)
        assert torch.isfinite(v_pppm).all()
        return v_pppm

    def velocities_x_to_fourmomenta(self, v_pppm, pppm, fourmomenta):
        px, py, pz, mass = unpack_last(pppm)
        E, px, py, pz = unpack_last(fourmomenta)
        v_px, v_py, v_pz, v_m = unpack_last(v_pppm)

        v_m *= self.mass_scale
        v_E = (mass * v_m + px * v_px + py * v_py + pz * v_pz) / E

        v_fourmomenta = torch.stack((v_E, v_px, v_py, v_pz), dim=-1)
        assert torch.isfinite(v_fourmomenta).all()
        return v_fourmomenta


class PPPM2(BaseCoordinates):
    def __init__(self, mass_scale=1.0):
        self.mass_scale = mass_scale

    def fourmomenta_to_x(self, fourmomenta):
        E, px, py, pz = unpack_last(fourmomenta)

        m2 = E**2 - (px**2 + py**2 + pz**2)
        m2 /= self.mass_scale**2

        pppm2 = torch.stack((px, py, pz, m2), dim=-1)
        assert torch.isfinite(pppm2).all()
        return pppm2

    def x_to_fourmomenta(self, pppm2):
        px, py, pz, m2 = unpack_last(pppm2)

        m2 *= self.mass_scale**2
        E = torch.sqrt(m2 + px**2 + py**2 + pz**2)

        fourmomenta = torch.stack((E, px, py, pz), dim=-1)
        assert torch.isfinite(fourmomenta).all()
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, pppm2):
        px, py, pz, m2 = unpack_last(pppm2)
        E, px, py, pz = unpack_last(fourmomenta)
        v_E, v_px, v_py, v_pz = unpack_last(v_fourmomenta)

        v_m2 = 2 * (E * v_E - px * v_px - py * v_py - pz * v_pz)
        v_m2 /= self.mass_scale**2

        v_pppm2 = torch.stack((v_px, v_py, v_pz, v_m2), dim=-1)
        assert torch.isfinite(v_pppm2).all()
        return v_pppm2

    def velocities_x_to_fourmomenta(self, v_pppm2, pppm2, fourmomenta):
        px, py, pz, m2 = unpack_last(pppm2)
        E, px, py, pz = unpack_last(fourmomenta)
        v_px, v_py, v_pz, v_m2 = unpack_last(v_pppm2)

        v_m2 *= self.mass_scale**2
        v_E = (v_m2 / 2 + px * v_px + py * v_py + pz * v_pz) / E

        v_fourmomenta = torch.stack((v_E, v_px, v_py, v_pz), dim=-1)
        assert torch.isfinite(v_fourmomenta).all()
        return v_fourmomenta


class PPPlogM2(PPPM2):
    def __init__(self, mass_scale=1.0):
        super().__init__(mass_scale=mass_scale)

    def fourmomenta_to_x(self, fourmomenta):
        pppm2 = super().fourmomenta_to_x(fourmomenta)
        px, py, pz, m2 = unpack_last(pppm2)

        logm2 = torch.log(m2.clamp(min=0) + EPS1)

        ppplogm2 = torch.stack((px, py, pz, logm2), dim=-1)
        assert torch.isfinite(ppplogm2).all()
        return ppplogm2

    def x_to_fourmomenta(self, ppplogm2):
        px, py, pz, logm2 = unpack_last(ppplogm2)

        m2 = logm2.clamp(max=CUTOFF).exp() - EPS1

        pppm2 = torch.stack((px, py, pz, m2), dim=-1)
        assert torch.isfinite(pppm2).all()

        fourmomenta = super().x_to_fourmomenta(pppm2)
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, ppplogm2):
        pppm2 = super().fourmomenta_to_x(fourmomenta)
        v_pppm2 = super().velocities_fourmomenta_to_x(v_fourmomenta, fourmomenta, pppm2)

        v_px, v_py, v_pz, v_m2 = unpack_last(v_pppm2)
        _, _, _, m2 = unpack_last(pppm2)
        v_logm2 = v_m2 / (m2 + EPS1)

        v_ppplogm2 = torch.stack((v_px, v_py, v_pz, v_logm2), dim=-1)
        assert torch.isfinite(v_ppplogm2).all()
        return v_ppplogm2

    def velocities_x_to_fourmomenta(self, v_ppplogm2, ppplogm2, fourmomenta):
        pppm2 = super().fourmomenta_to_x(fourmomenta)
        px, py, pz, logm2 = unpack_last(ppplogm2)
        px, py, pz, m2 = unpack_last(pppm2)
        E, px, py, pz = unpack_last(fourmomenta)
        v_px, v_py, v_pz, v_logm2 = unpack_last(v_ppplogm2)

        v_m2 = (m2 + EPS1) * v_logm2

        v_pppm2 = torch.stack((v_px, v_py, v_pz, v_m2), dim=-1)
        assert torch.isfinite(v_pppm2).all()

        v_fourmomenta = super().velocities_x_to_fourmomenta(v_pppm2, pppm2, fourmomenta)
        return v_fourmomenta


class Jetmomenta(BaseCoordinates):
    def fourmomenta_to_x(self, fourmomenta):
        jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
        return jetmomenta

    def x_to_fourmomenta(self, jetmomenta):
        fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, jetmomenta):
        pt, phi, eta, mass = unpack_last(jetmomenta)
        E, px, py, pz = unpack_last(fourmomenta)
        v_E, v_px, v_py, v_pz = unpack_last(v_fourmomenta)

        v_pt = (px * v_px + py * v_py) / pt
        v_phi = (px * v_py - py * v_px) / pt**2
        v_eta = (
            (pt**2 * v_pz - pz * (px * v_px + py * v_py))
            / pt**2
            / torch.sqrt(px**2 + py**2 + pz**2)
        )
        v_mass = (E * v_E - px * v_px - py * v_py - pz * v_pz) / mass.clamp(min=1e-5)

        v_jetmomenta = torch.stack((v_pt, v_phi, v_eta, v_mass), dim=-1)
        assert torch.isfinite(v_jetmomenta).all()
        return v_jetmomenta

    def velocities_x_to_fourmomenta(self, v_jetmomenta, jetmomenta, fourmomenta):
        pt, phi, eta, mass = unpack_last(jetmomenta)
        E, px, py, pz = unpack_last(fourmomenta)
        v_pt, v_phi, v_eta, v_mass = unpack_last(v_jetmomenta)

        v_px = torch.cos(phi) * v_pt - torch.sin(phi) * pt * v_phi
        v_py = torch.sin(phi) * v_pt + torch.cos(phi) * pt * v_phi
        v_pz = torch.cosh(eta) * pt * v_eta + torch.sinh(eta) * v_pt
        v_E = (
            mass * v_mass
            + torch.cosh(eta) ** 2 * pt * v_pt
            + torch.cosh(eta) * torch.sinh(eta) * pt**2 * v_eta
        ) / E

        v_fourmomenta = torch.stack((v_E, v_px, v_py, v_pz), dim=-1)
        assert torch.isfinite(v_fourmomenta).all()
        return v_fourmomenta

    def final_checks(self, jetmomenta):
        jetmomenta[..., 1] = ensure_angle(jetmomenta[..., 1])
        return jetmomenta


class Precisesiast(Jetmomenta):
    def __init__(self, pt_min):
        self.pt_min = pt_min

    def fourmomenta_to_x(self, fourmomenta):
        jetmomenta = super().fourmomenta_to_x(fourmomenta)
        pt, phi, eta, mass = unpack_last(jetmomenta)

        pt_min_local = self.pt_min[:, : pt.shape[-1]].clone().to(pt.device)
        x_pt = torch.log(pt - pt_min_local + EPS1)
        x_mass = torch.log(mass + EPS1)

        precisesiast = torch.stack((x_pt, phi, eta, x_mass), dim=-1)
        assert torch.isfinite(precisesiast).all()
        return precisesiast

    def x_to_fourmomenta(self, precisesiast):
        x_pt, phi, eta, x_mass = unpack_last(precisesiast)

        pt_min_local = self.pt_min[:, : x_pt.shape[-1]].clone().to(x_pt.device)
        pt = x_pt.clamp(max=CUTOFF).exp() + pt_min_local - EPS1
        mass = x_mass.clamp(max=CUTOFF).exp() - EPS1

        jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
        assert torch.isfinite(jetmomenta).all()

        fourmomenta = super().x_to_fourmomenta(jetmomenta)
        return fourmomenta

    def velocities_fourmomenta_to_x(self, v_fourmomenta, fourmomenta, precisesiast):
        jetmomenta = super().fourmomenta_to_x(fourmomenta)
        v_jetmomenta = super().velocities_fourmomenta_to_ - x(
            v_fourmomenta, fourmomenta, jetmomenta
        )

        x_pt, x_phi, eta, mass = unpack_last(precisesiast)
        pt, phi, eta, mass = unpack_last(jetmomenta)
        v_pt, v_phi, v_eta, v_mass = unpack_last(v_jetmomenta)

        pt_min_local = self.pt_min[:, : x_pt.shape[-1]].clone().to(x_pt.device)
        v_x_pt = v_pt / (pt - pt_min_local)
        v_x_mass = v_mass / (mass + EPS1)

        v_precisesiast = torch.stack((v_x_pt, v_phi, v_eta, v_x_mass), dim=-1)
        assert torch.isfinite(v_precisesiast).all()
        return v_precisesiast

    def velocities_x_to_fourmomenta(self, v_precisesiast, precisesiast, fourmomenta):
        jetmomenta = super().fourmomenta_to_x(fourmomenta)

        x_pt, x_phi, eta, mass = unpack_last(precisesiast)
        pt, phi, eta, mass = unpack_last(jetmomenta)
        v_x_pt, v_phi, v_eta, v_x_mass = unpack_last(v_precisesiast)

        v_pt = pt * v_x_pt
        v_mass = mass * v_x_mass

        v_jetmomenta = torch.stack((v_pt, v_phi, v_eta, v_mass), dim=-1)
        assert torch.isfinite(v_jetmomenta).all()

        v_fourmomenta = super().velocities_x_to_fourmomenta(
            v_jetmomenta, jetmomenta, fourmomenta
        )
        return v_jetmomenta
