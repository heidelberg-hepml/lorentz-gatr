import torch

from experiments.eventgen.physics import (
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
    jetmomenta_to_precisesiast,
    precisesiast_to_jetmomenta,
)


def ensure_onshell(fourmomenta, onshell_list, onshell_mass):
    masses = torch.tensor(
        onshell_mass, device=fourmomenta.device, dtype=fourmomenta.dtype
    )
    masses = masses.unsqueeze(0).expand(fourmomenta.shape[0], masses.shape[-1])
    fourmomenta[..., onshell_list, 0] = torch.sqrt(
        masses**2 + torch.sum(fourmomenta[..., onshell_list, 1:] ** 2, dim=-1)
    )
    return fourmomenta


def preprocess_gatr(fourmomenta, pt_min, prep_params=None):
    if prep_params is None:
        prep_params = {"std": fourmomenta.std()}
    fourmomenta = fourmomenta / prep_params["std"]
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    precisesiast = jetmomenta_to_precisesiast(
        jetmomenta, pt_min, for_gatr=True, prep_params=prep_params
    )
    return precisesiast, prep_params


def undo_preprocess_gatr(precisesiast, pt_min, prep_params):
    jetmomenta = precisesiast_to_jetmomenta(
        precisesiast, pt_min, for_gatr=True, prep_params=prep_params
    )
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
    fourmomenta = fourmomenta * prep_params["std"]
    return fourmomenta


def preprocess_tr(fourmomenta, pt_min, prep_params=None):
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    precisesiast = jetmomenta_to_precisesiast(jetmomenta, pt_min, for_gatr=False)
    if prep_params is None:
        prep_params = {
            "std": precisesiast.std(dim=[0, 1], keepdim=True),
            "mean": precisesiast.mean(dim=[0, 1], keepdim=True),
        }
    precisesiast = (precisesiast - prep_params["mean"]) / prep_params["std"]
    return precisesiast, prep_params


def undo_preprocess_tr(precisesiast, pt_min, prep_params):
    precisesiast = precisesiast * prep_params["std"] + prep_params["mean"]
    jetmomenta = precisesiast_to_jetmomenta(precisesiast, pt_min, for_gatr=False)
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
    return fourmomenta
