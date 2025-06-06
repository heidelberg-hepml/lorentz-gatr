import pytest
import torch

import experiments.eventgen.coordinates as c
from experiments.eventgen.distributions import (
    NaivePPPM2,
    NaivePPPLogM2,
    StandardPPPLogM2,
    StandardLogPtPhiEtaLogM2,
)
from experiments.eventgen.processes import ttbarExperiment
from tests.helpers import TOLERANCES


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.StandardPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.StandardLogPtPhiEtaLogM2,
        c.PtPhiEtaM2Relative,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        NaivePPPM2,
        NaivePPPLogM2,
        StandardPPPLogM2,
        StandardLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [1000])
def test_invertibility(coordinates, distribution, experiment_np, nevents):
    """test invertibility of forward() and inverse() methods"""
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    d.coordinates.init_unit([nparticles])
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32
    if coordinates in [
        c.StandardLogPtPhiEtaLogM2,
        c.LogPtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaE,
    ]:
        coord = coordinates(exp.pt_min, exp.units)
    elif coordinates == c.StandardPPPLogM2:
        coord = coordinates(exp.onshell_list)
    elif coordinates in [c.StandardLogPtPhiEtaLogM2]:
        coord = coordinates(exp.pt_min, exp.units, exp.onshell_list)
    else:
        coord = coordinates()

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)

    kwargs = {}
    if coordinates == c.PtPhiEtaM2Relative:
        coordinates = c.PtPhiEtaM2()
        jet = fourmomenta_original.sum(dim=-2, keepdim=True).expand(
            fourmomenta_original.shape
        )
        jet_x = coordinates.fourmomenta_to_x(jet)
        kwargs["jet"] = jet_x

    # init_fit (this does nothing, except for FitNormal)
    coord.init_fit([fourmomenta_original], **kwargs)

    # forward and inverse transform
    x_original = coord.fourmomenta_to_x(fourmomenta_original, **kwargs)
    fourmomenta_transformed = coord.x_to_fourmomenta(x_original, **kwargs)
    x_transformed = coord.fourmomenta_to_x(fourmomenta_transformed, **kwargs)

    torch.testing.assert_close(
        fourmomenta_original, fourmomenta_transformed, **TOLERANCES
    )
    torch.testing.assert_close(x_original, x_transformed, **TOLERANCES)


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.StandardPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.StandardLogPtPhiEtaLogM2,
        c.PtPhiEtaM2Relative,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        NaivePPPM2,
        NaivePPPLogM2,
        StandardPPPLogM2,
        StandardLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [1000])
def test_velocity(coordinates, distribution, experiment_np, nevents):
    """test correctness of jacobians from _jac_forward() and _jac_inverse() methods, and their invertibility"""
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    d.coordinates.init_unit([nparticles])
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32
    if coordinates in [
        c.StandardLogPtPhiEtaLogM2,
        c.LogPtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaE,
    ]:
        coord = coordinates(exp.pt_min, exp.units)
    else:
        coord = coordinates()

    shape = (nevents, nparticles, 4)
    x = d.sample(shape, device, dtype)

    kwargs = {}
    if coordinates == c.PtPhiEtaM2Relative:
        coordinates = c.PtPhiEtaM2()
        jet = x.sum(dim=-2, keepdim=True).expand(x.shape)
        jet_x = coordinates.fourmomenta_to_x(jet)
        kwargs["jet"] = jet_x

    # init_fit (this does nothing, except for FitNormal)
    coord.init_fit([x], **kwargs)

    x.requires_grad_()
    v_x = torch.randn_like(x)
    v_y, y = coord.velocity_fourmomenta_to_x(v_x, x, **kwargs)
    v_z, z = coord.velocity_x_to_fourmomenta(v_y, y, **kwargs)

    # jacobians from autograd
    jac_fw_autograd, jac_inv_autograd = [], []
    for i in range(4):
        grad_outputs = torch.zeros_like(x)
        grad_outputs[..., i] = 1.0
        fw_autograd = torch.autograd.grad(
            y,
            x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        inv_autograd = torch.autograd.grad(
            z,
            y,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        jac_fw_autograd.append(fw_autograd)
        jac_inv_autograd.append(inv_autograd)
    jac_fw_autograd = torch.stack(jac_fw_autograd, dim=-2)
    jac_inv_autograd = torch.stack(jac_inv_autograd, dim=-2)

    v_y_autograd = torch.einsum("...ij,...j->...i", jac_fw_autograd, v_x)
    v_z_autograd = torch.einsum("...ij,...j->...i", jac_inv_autograd, v_y)

    # compare to autograd
    torch.testing.assert_close(v_y, v_y_autograd, **TOLERANCES)
    torch.testing.assert_close(v_z, v_z_autograd, **TOLERANCES)


@pytest.mark.parametrize(
    "coordinates",
    [
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.StandardPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.StandardLogPtPhiEtaLogM2,
        c.PtPhiEtaM2Relative,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        NaivePPPM2,
        NaivePPPLogM2,
        StandardPPPLogM2,
        StandardLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [1000])
def test_logdetjac(coordinates, distribution, experiment_np, nevents):
    """test correctness of jacobians from logdetjac_fourmomenta_to_x() and logdetjac_x_to_fourmomenta() methods, and their invertibility"""
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    d.coordinates.init_unit([nparticles])
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32
    if coordinates in [
        c.StandardLogPtPhiEtaLogM2,
        c.LogPtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaE,
    ]:
        coord = coordinates(exp.pt_min, exp.units)
    else:
        coord = coordinates()

    shape = (nevents, nparticles, 4)
    x = d.sample(shape, device, dtype)

    kwargs = {}
    if coordinates == c.PtPhiEtaM2Relative:
        coordinates = c.PtPhiEtaM2()
        jet = x.sum(dim=-2, keepdim=True).expand(x.shape)
        jet_x = coordinates.fourmomenta_to_x(jet)
        kwargs["jet"] = jet_x

    # init_fit (this does nothing, except for FitNormal)
    coord.init_fit([x], **kwargs)

    x.requires_grad_()
    logdetjac_fw, y = coord.logdetjac_fourmomenta_to_x(x, **kwargs)
    logdetjac_inv, z = coord.logdetjac_x_to_fourmomenta(y, **kwargs)

    # jacobians from autograd
    jac_fw_autograd, jac_inv_autograd = [], []
    for i in range(4):
        grad_outputs = torch.zeros_like(x)
        grad_outputs[..., i] = 1.0
        fw_autograd = torch.autograd.grad(
            y,
            x,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        inv_autograd = torch.autograd.grad(
            z,
            y,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        jac_fw_autograd.append(fw_autograd)
        jac_inv_autograd.append(inv_autograd)
    jac_fw_autograd = torch.stack(jac_fw_autograd, dim=-2)
    jac_inv_autograd = torch.stack(jac_inv_autograd, dim=-2)

    logdetjac_fw_autograd = (
        -torch.linalg.det(jac_fw_autograd).abs().log().sum(dim=-1, keepdims=True)
    )
    logdetjac_inv_autograd = (
        -torch.linalg.det(jac_inv_autograd).abs().log().sum(dim=-1, keepdims=True)
    )

    # compare to autograd
    torch.testing.assert_close(logdetjac_fw, logdetjac_fw_autograd, **TOLERANCES)
    torch.testing.assert_close(logdetjac_inv, logdetjac_inv_autograd, **TOLERANCES)
