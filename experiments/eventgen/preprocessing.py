import torch

from experiments.eventgen.transforms import (
    fourmomenta_to_jetmomenta,
    jetmomenta_to_fourmomenta,
    jetmomenta_to_precisesiast,
    precisesiast_to_jetmomenta,
    get_mass,
)


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


def preprocess(fourmomenta, pt_min, is_gatr, prep_params):
    fourmomenta = fourmomenta / prep_params["units"]
    jetmomenta = fourmomenta_to_jetmomenta(fourmomenta)
    precisesiast = jetmomenta_to_precisesiast(jetmomenta, pt_min)
    if not is_gatr:
        # standardize components
        if (
            prep_params.get("std", None) is None
            or prep_params.get("mean", None) is None
        ):
            prep_params["std"] = precisesiast.std(dim=[0, 1], keepdim=True)
            prep_params["mean"] = precisesiast.mean(dim=[0, 1], keepdim=True)
        precisesiast = (precisesiast - prep_params["mean"]) / prep_params["std"]
    return precisesiast, prep_params


def undo_preprocess(precisesiast, pt_min, is_gatr, prep_params):
    if not is_gatr:
        precisesiast = precisesiast * prep_params["std"] + prep_params["mean"]
    jetmomenta = precisesiast_to_jetmomenta(precisesiast, pt_min)
    fourmomenta = jetmomenta_to_fourmomenta(jetmomenta)
    fourmomenta = fourmomenta * prep_params["units"]
    return fourmomenta
