import math
import torch

# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-2

# generic numerical stability cutoff
EPS2 = 1e-10

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 7


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta):
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = torch.stack((E, px, py, pz), dim=-1)
    assert torch.isfinite(fourmomenta).all()
    return fourmomenta


def jetmomenta_to_precisesiast(jetmomenta, pt_min):
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))

    pt_min_local = pt_min[:, : pt.shape[-1]].clone().to(pt.device)
    x_pt = torch.log(pt - pt_min_local + EPS1)
    x_mass = torch.log(mass + EPS1)

    precisesiast = torch.stack((x_pt, phi, eta, x_mass), dim=-1)
    assert torch.isfinite(precisesiast).all()
    return precisesiast


def precisesiast_to_jetmomenta(precisesiast, pt_min):
    x_pt, phi, eta, x_mass = torch.permute(precisesiast, (2, 0, 1))

    pt_min_local = pt_min[:, : x_pt.shape[-1]].clone().to(x_pt.device)
    pt = x_pt.clamp(max=CUTOFF).exp() + pt_min_local - EPS1
    mass = x_mass.clamp(max=CUTOFF).exp() - EPS1

    jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(jetmomenta).all()
    return jetmomenta


def velocities_jetmomenta_to_fourmomenta(v_jetmomenta, jetmomenta, fourmomenta):
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))
    E, px, py, pz = torch.permute(fourmomenta, (2, 0, 1))
    v_pt, v_phi, v_eta, v_mass = torch.permute(v_jetmomenta, (2, 0, 1))

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


def velocities_precisesiast_to_jetmomenta(v_precisesiast, precisesiast, jetmomenta):
    x_pt, x_phi, eta, mass = torch.permute(precisesiast, (2, 0, 1))
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))
    v_x_pt, v_phi, v_eta, v_x_mass = torch.permute(v_precisesiast, (2, 0, 1))

    v_pt = pt * v_x_pt
    v_mass = mass * v_x_mass

    v_jetmomenta = torch.stack((v_pt, v_phi, v_eta, v_mass), dim=-1)
    assert torch.isfinite(v_jetmomenta).all()
    return v_jetmomenta


def velocities_fourmomenta_to_jetmomenta(v_fourmomenta, fourmomenta, jetmomenta):
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))
    E, px, py, pz = torch.permute(fourmomenta, (2, 0, 1))
    v_E, v_px, v_py, v_pz = torch.permute(v_fourmomenta, (2, 0, 1))

    v_pt = (px * v_pz + py * v_py) / pt
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


def velocities_jetmomenta_to_precisesiast(
    v_jetmomenta, jetmomenta, precisesiast, pt_min
):
    x_pt, x_phi, eta, mass = torch.permute(precisesiast, (2, 0, 1))
    pt, phi, eta, mass = torch.permute(jetmomenta, (2, 0, 1))
    v_pt, v_phi, v_eta, v_mass = torch.permute(v_jetmomenta, (2, 0, 1))

    pt_min_local = pt_min[:, : x_pt.shape[-1]].clone().to(x_pt.device)
    v_x_pt = v_pt / (pt - pt_min_local)
    v_x_mass = v_mass / mass.clamp(min=1e-5)

    v_precisesiast = torch.stack((v_x_pt, v_phi, v_eta, v_x_mass), dim=-1)
    assert torch.isfinite(v_precisesiast).all()
    return v_precisesiast


def get_pt(particle):
    return torch.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return torch.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle):
    p_abs = torch.sqrt(torch.sum(particle[..., 1:] ** 2, dim=-1))
    eta = stable_arctanh(particle[..., 3] / p_abs, eps=EPS2)
    return eta


def stable_arctanh(x, eps=EPS2):
    # numerically stable implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def get_mass(particle, eps=EPS2):
    return torch.sqrt(
        torch.clamp(
            particle[..., 0] ** 2 - torch.sum(particle[..., 1:] ** 2, dim=-1), min=EPS2
        )
    )


def ensure_angle(phi):
    return (phi + math.pi) % (2 * math.pi) - math.pi


def ensure_onshell(fourmomenta, onshell_list, onshell_mass):
    onshell_mass = torch.tensor(
        onshell_mass, device=fourmomenta.device, dtype=fourmomenta.dtype
    )
    onshell_mass = onshell_mass.unsqueeze(0).expand(
        fourmomenta.shape[0], onshell_mass.shape[-1]
    )
    fourmomenta[..., onshell_list, 0] = torch.sqrt(
        onshell_mass**2 + torch.sum(fourmomenta[..., onshell_list, 1:] ** 2, dim=-1)
    )
    return fourmomenta


def delta_phi(event, idx1, idx2, abs=False):
    dphi = event[..., idx1, 1] - event[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return torch.abs(dphi) if abs else dphi


def delta_eta(event, idx1, idx2, abs=False):
    deta = event[..., idx1, 2] - event[..., idx2, 2]
    return torch.abs(deta) if abs else deta


def delta_r(event, idx1, idx2):
    return (
        delta_phi(event, idx1, idx2) ** 2 + delta_eta(event, idx1, idx2) ** 2
    ) ** 0.5


def get_virtual_particle(event, components):
    jetmomenta = event.clone()
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

    particle = fourmomenta[..., components, :].sum(dim=-2)
    particle = fourmomenta_to_jetmomenta(particle)
    return particle
