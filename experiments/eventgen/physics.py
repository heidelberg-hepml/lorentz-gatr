import numpy as np


def fourmomenta_to_jetmomenta(fourmomenta):
    pt = get_pt(fourmomenta)
    phi = get_phi(fourmomenta)
    eta = get_eta(fourmomenta)
    mass = get_mass(fourmomenta)

    jetmomenta = np.stack((pt, phi, eta, mass), axis=-1)
    assert np.isfinite(jetmomenta).all()
    return jetmomenta


def jetmomenta_to_fourmomenta(jetmomenta, cutoff=10):
    pt, phi, eta, mass = jetmomenta.transpose(2, 0, 1)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(np.clip(eta, -cutoff, cutoff))
    E = np.sqrt(mass**2 + px**2 + py**2 + pz**2)

    fourmomenta = np.stack((E, px, py, pz), axis=-1)
    assert np.isfinite(fourmomenta).all()
    return fourmomenta


def get_pt(particle):
    return np.sqrt(particle[..., 1] ** 2 + particle[..., 2] ** 2)


def get_phi(particle):
    return np.arctan2(particle[..., 2], particle[..., 1])


def get_eta(particle, eps=1e-10):
    # eta = np.arctanh(particle[...,3] / p_abs) # numerically unstable
    p_abs = np.sqrt(np.sum(particle[..., 1:] ** 2, axis=-1))
    eta = 0.5 * (
        np.log(np.clip(np.abs(p_abs + particle[..., 3]), eps, None))
        - np.log(np.clip(np.abs(p_abs - particle[..., 3]), eps, None))
    )
    return eta


def get_mass(particle, eps=1e-6):
    return np.sqrt(
        np.clip(
            particle[..., 0] ** 2 - np.sum(particle[..., 1:] ** 2, axis=-1), eps, None
        )
    )


def ensure_angle(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def delta_phi(event, idx1, idx2, abs=False):
    dphi = event[..., idx1, 1] - event[..., idx2, 1]
    dphi = ensure_angle(dphi)
    return np.abs(dphi) if abs else dphi


def delta_eta(event, idx1, idx2, abs=False):
    deta = event[..., idx1, 2] - event[..., idx2, 2]
    return np.abs(deta) if abs else deta


def delta_r(event, idx1, idx2):
    return (
        delta_phi(event, idx1, idx2) ** 2 + delta_eta(event, idx1, idx2) ** 2
    ) ** 0.5


def get_virtual_particle(event, components):
    jetmomenta = event.copy()
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)

    particle = fourmomenta[..., components, :].sum(axis=-2)
    particle = fourmomenta_to_jetmomenta(particle)
    return particle
