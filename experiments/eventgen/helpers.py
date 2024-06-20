import math
import torch

# log(x) -> log(x+EPS1)
# in (invertible) preprocessing functions to avoid being close to log(0)
EPS1 = 1e-3

# generic numerical stability cutoff
EPS2 = 1e-10

# exp(x) -> exp(x.clamp(max=CUTOFF))
CUTOFF = 10

# these functions are only used for plotting,
# with the exception of delta_r_fast


def unpack_last(x):
    # unpack along the last dimension
    n = len(x.shape)
    return torch.permute(x, (n - 1, *list(range(n - 1))))


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = torch.stack((pt, phi, eta, mass), dim=-1)
    assert torch.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta):
    pt, phi, eta, mass = unpack_last(jetmomenta)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta.clamp(min=-CUTOFF, max=CUTOFF))
    E = torch.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = torch.stack((E, px, py, pz), dim=-1)
    assert torch.isfinite(fourmomenta).all()
    return fourmomenta


def stay_positive(x):
    # flip sign for entries with x<0 such that always x>0
    x = torch.where(x >= 0, x, -x)
    return x


def get_pt(fourmomenta):
    return torch.sqrt(fourmomenta[..., 1] ** 2 + fourmomenta[..., 2] ** 2)


def get_phi(fourmomenta):
    return torch.arctan2(fourmomenta[..., 2], fourmomenta[..., 1])


def get_eta(fourmomenta):
    p_abs = torch.sqrt(torch.sum(fourmomenta[..., 1:] ** 2, dim=-1))
    eta = stable_arctanh(fourmomenta[..., 3] / p_abs, eps=EPS2)
    return eta


def stable_arctanh(x, eps=EPS2):
    # implementation of arctanh that avoids log(0) issues
    return 0.5 * (torch.log((1 + x).clamp(min=eps)) - torch.log((1 - x).clamp(min=eps)))


def get_mass(fourmomenta, eps=EPS2):
    m2 = fourmomenta[..., 0] ** 2 - torch.sum(fourmomenta[..., 1:] ** 2, dim=-1)
    m2 = stay_positive(m2)
    m = torch.sqrt(m2.clamp(min=EPS2))
    return m


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


def delta_phi(jetmomenta, idx1, idx2, abs=False):
    dphi = jetmomenta[..., idx1, 1] - jetmomenta[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return torch.abs(dphi) if abs else dphi


def delta_eta(jetmomenta, idx1, idx2, abs=False):
    deta = jetmomenta[..., idx1, 2] - jetmomenta[..., idx2, 2]
    return torch.abs(deta) if abs else deta


def delta_r(jetmomenta, idx1, idx2):
    return (
        delta_phi(jetmomenta, idx1, idx2) ** 2 + delta_eta(jetmomenta, idx1, idx2) ** 2
    ) ** 0.5


def delta_r_fast(jetmomenta1, jetmomenta2):
    dphi = ensure_angle(jetmomenta1[..., 1] - jetmomenta2[..., 1])
    deta = jetmomenta1[..., 2] - jetmomenta2[..., 2]
    return (dphi**2 + deta**2) ** 0.5


def get_virtual_particle(jetmomenta, components):
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

    particle = fourmomenta[..., components, :].sum(dim=-2)
    particle = fourmomenta_to_jetmomenta(particle)
    return particle
