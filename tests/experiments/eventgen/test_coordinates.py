import pytest
import torch
import numpy as np

import experiments.eventgen.coordinates as c
from experiments.eventgen.distributions import (
    StandardPPPM2,
    StandardPPPLogM2,
    FittedPPPLogM2,
    FittedLogPtPhiEtaLogM2,
)
from experiments.eventgen.ttbarexperiment import ttbarExperiment
from experiments.eventgen.zmumuexperiment import zmumuExperiment
from tests.helpers import STRICT_TOLERANCES as TOLERANCES


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.FittedPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        StandardPPPM2,
        StandardPPPLogM2,
        FittedPPPLogM2,
        FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
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
        c.FittedLogPtPhiEtaLogM2,
        c.LogPtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaE,
    ]:
        coord = coordinates(exp.pt_min, exp.units)
    else:
        coord = coordinates()

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)

    # init_fit (this does nothing, except for FitNormal)
    coord.init_fit([fourmomenta_original])

    # forward and inverse transform
    x_original = coord.fourmomenta_to_x(fourmomenta_original)
    fourmomenta_transformed = coord.x_to_fourmomenta(x_original)
    x_transformed = coord.fourmomenta_to_x(fourmomenta_transformed)

    torch.testing.assert_close(
        fourmomenta_original, fourmomenta_transformed, **TOLERANCES
    )  # runs fine
    # print(((x_original - x_transformed).abs() > 0.1).sum(dim=[0,1]))
    # mask = ((x_original - x_transformed).abs() > 0.1).any(dim=[1,2])
    # print(x_original[mask,:,3], x_transformed[mask,:,3])
    torch.testing.assert_close(
        x_original, x_transformed, **TOLERANCES
    )  # fails with 0.1% rate for Fitted + zmumu (masses)


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PPPM2,
        c.EPhiPtPz,
        c.PtPhiEtaE,
        c.PtPhiEtaM2,
        c.PPPLogM2,
        c.FittedPPPLogM2,
        c.LogPtPhiEtaE,
        c.PtPhiEtaLogM2,
        c.LogPtPhiEtaM2,
        c.LogPtPhiEtaLogM2,
        c.FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        StandardPPPM2,
        StandardPPPLogM2,
        FittedPPPLogM2,
        FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
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
        c.FittedLogPtPhiEtaLogM2,
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

    # init_fit (this does nothing, except for FitNormal)
    coord.init_fit([x])

    x.requires_grad_()
    y = coord.fourmomenta_to_x(x)
    z = coord.x_to_fourmomenta(y)
    v_x = torch.randn_like(x)
    v_y = coord.velocity_fourmomenta_to_x(v_x, x)
    v_z = coord.velocity_x_to_fourmomenta(v_y, y)

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
    # print(v_y[0, 0, ...], v_y_autograd[0, 0, ...], jac_fw_autograd[0, 0, ...])
    # print(((v_y - v_y_autograd).abs() > 0.1).sum(dim=[0, 1]))
    torch.testing.assert_close(v_y, v_y_autograd, **TOLERANCES)
    # print(v_z[0,0,...], v_z_autograd[0,0,...], jac_inv_autograd[0,0,...])
    # print(((v_z - v_z_autograd).abs() > 0.1).sum(dim=[0,1]))
    torch.testing.assert_close(v_z, v_z_autograd, **TOLERANCES)