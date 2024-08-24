import pytest
import torch
import numpy as np

import experiments.eventgen.transforms as tr
from experiments.eventgen.distributions import (
    NaivePPPM2,
    NaivePPPLogM2,
    StandardPPPLogM2,
    StandardLogPtPhiEtaLogM2,
)
from experiments.eventgen.processes import ttbarExperiment, zmumuExperiment
from tests.helpers import STRICT_TOLERANCES as TOLERANCES


def test_simple():
    """Some very simple tests"""
    fourmomentum = torch.tensor([[1, 1, 0, 0], [1, 1, 0, -1]]).float()
    ptphietam2 = torch.tensor(
        [[1, 0, 0, 0], [1, 0, np.arctanh(-1 / 2**0.5), 1]]
    ).float()
    transforms = [tr.EPPP_to_PtPhiEtaE(), tr.PtPhiEtaE_to_PtPhiEtaM2()]
    x = fourmomentum.clone()
    for t in transforms:
        x = t.forward(x)
    torch.testing.assert_close(x, ptphietam2, **TOLERANCES)


@pytest.mark.parametrize(
    "transforms",
    [
        [tr.StandardNormal],
        [tr.EPPP_to_PPPM2],
        [tr.EPPP_to_EPhiPtPz],
        [tr.EPPP_to_PtPhiEtaE],
        [tr.EPPP_to_PPPM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.Pt_to_LogPt],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.Pt_to_LogPt],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
        ],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
            tr.StandardNormal,
        ],
        [tr.EPPP_to_EPhiPtPz, tr.NonPeriodicPhi],
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
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 3], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
def test_invertibility(transforms, distribution, experiment_np, nevents):
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
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_LogPt:
            ts.append(tra(exp.pt_min, exp.units))
        elif tra == tr.StandardNormal:
            local = tra([0, 1, 2, 3])
            local.init_unit([nparticles])
            ts.append(local)
        else:
            ts.append(tra())

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)
    x = fourmomenta_original.clone()

    # init_fit (has to be done manually in this case
    # this does nothing, except for StandardNormal
    x_fit = x.clone()
    for t in ts[:-1]:
        x_fit = t.forward(x)
    ts[-1].init_fit([x])

    for t in ts:
        x = t.forward(x)
    x_original = x.clone()
    for t in ts[::-1]:
        x = t.inverse(x)
    fourmomenta_transformed = x.clone()
    for t in ts:
        x = t.forward(x)
    x_transformed = x.clone()

    torch.testing.assert_close(
        fourmomenta_original, fourmomenta_transformed, **TOLERANCES
    )
    torch.testing.assert_close(x_original, x_transformed, **TOLERANCES)


@pytest.mark.parametrize(
    "transforms",
    [
        [tr.StandardNormal],
        [tr.EPPP_to_PPPM2],
        [tr.EPPP_to_EPhiPtPz],
        [tr.EPPP_to_PtPhiEtaE],
        [tr.EPPP_to_PPPM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.Pt_to_LogPt],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.Pt_to_LogPt],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
        ],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
            tr.StandardNormal,
        ],
        [tr.EPPP_to_EPhiPtPz, tr.NonPeriodicPhi],
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
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 3], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
def test_jacobians(transforms, distribution, experiment_np, nevents):
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
    dtype = torch.float64  # sometimes fails with torch32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_LogPt:
            ts.append(tra(exp.pt_min, exp.units))
        elif tra == tr.StandardNormal:
            local = tra([0, 1, 2, 3])
            local.init_unit([nparticles])
            ts.append(local)
        else:
            ts.append(tra())

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)
    x = fourmomenta_original.clone()

    # init_fit (has to be done manually in this case
    # this does nothing, except for StandardNormal
    x_fit = x.clone()
    for t in ts[:-1]:
        x_fit = t.forward(x)
    ts[-1].init_fit([x])

    x.requires_grad_()
    xs = [x.clone()]
    for t in ts[:-1]:
        x = t.forward(x)
        xs.append(x.clone())
    x = xs[-1]
    y = ts[-1].forward(x)
    z = ts[-1].inverse(y)

    # jacobians from code
    jac_fw = ts[-1]._jac_forward(x, y)
    jac_inv = ts[-1]._jac_inverse(y, z)

    # test jacobian invertibility
    diag_left = torch.einsum("...ij,...jk->...ik", jac_fw, jac_inv)
    diag_right = torch.einsum("...ij,...jk->...ik", jac_inv, jac_fw)
    diag = torch.eye(4, dtype=dtype)[None, None, ...].expand(diag_left.shape)
    torch.testing.assert_close(diag_right, diag, **TOLERANCES)
    torch.testing.assert_close(diag_left, diag, **TOLERANCES)

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

    # compare jacobian to autograd
    torch.testing.assert_close(jac_fw, jac_fw_autograd, **TOLERANCES)
    torch.testing.assert_close(jac_inv, jac_inv_autograd, **TOLERANCES)


@pytest.mark.parametrize(
    "transforms",
    [
        [tr.StandardNormal],
        [tr.EPPP_to_PPPM2],
        [tr.EPPP_to_EPhiPtPz],
        [tr.EPPP_to_PtPhiEtaE],
        [tr.EPPP_to_PPPM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.Pt_to_LogPt],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.M2_to_LogM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.Pt_to_LogPt],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
        ],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_LogM2,
            tr.Pt_to_LogPt,
            tr.StandardNormal,
        ],
        [tr.EPPP_to_EPhiPtPz, tr.NonPeriodicPhi],
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
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 3], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
def test_logdetjac(transforms, distribution, experiment_np, nevents):
    """compare logdetjac_forward and logdetjac_inverse methods to autograd"""
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
    dtype = torch.float64  # sometimes fails with torch32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_LogPt:
            ts.append(tra(exp.pt_min, exp.units))
        elif tra == tr.StandardNormal:
            local = tra([0, 1, 2, 3])
            local.init_unit([nparticles])
            ts.append(local)
        else:
            ts.append(tra())

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)
    x = fourmomenta_original.clone()

    # init_fit (has to be done manually in this case
    # this does nothing, except for StandardNormal
    x_fit = x.clone()
    for t in ts[:-1]:
        x_fit = t.forward(x)
    ts[-1].init_fit([x])

    x.requires_grad_()
    xs = [x.clone()]
    for t in ts[:-1]:
        x = t.forward(x)
        xs.append(x.clone())
    x = xs[-1]
    y = ts[-1].forward(x)
    z = ts[-1].inverse(y)

    # logdetjac from code
    logdetjac_fw = ts[-1].logdetjac_forward(x, y)
    logdetjac_inv = ts[-1].logdetjac_inverse(y, z)

    # logdetjac from autograd
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
        torch.linalg.det(jac_fw_autograd).abs().log().sum(dim=-1, keepdims=True)
    )
    logdetjac_inv_autograd = (
        torch.linalg.det(jac_inv_autograd).abs().log().sum(dim=-1, keepdims=True)
    )

    # compare logdetjac to autograd
    torch.testing.assert_close(logdetjac_fw, logdetjac_fw_autograd, **TOLERANCES)
    torch.testing.assert_close(logdetjac_inv, logdetjac_inv_autograd, **TOLERANCES)
