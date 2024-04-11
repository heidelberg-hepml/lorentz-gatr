import pytest
import torch
import numpy as np

import experiments.eventgen.transforms as tr
from experiments.eventgen.distributions import (
    NaiveDistribution,
    NaiveLogDistribution,
    FourmomentaDistribution,
    JetmomentaDistribution,
)
from experiments.eventgen.ttbarexperiment import ttbarExperiment
from experiments.eventgen.zmumuexperiment import zmumuExperiment
from tests.helpers import MILD_TOLERANCES as TOLERANCES


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
        [tr.EPPP_to_PPPM2],
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
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        NaiveDistribution,
        NaiveLogDistribution,
        FourmomentaDistribution,
        JetmomentaDistribution,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
def test_invertibility(transforms, distribution, experiment_np, nevents):
    """test forward() and inverse() methods"""
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.base_kwargs,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with float32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_LogPt:
            ts.append(tra(exp.pt_min, exp.units))
        else:
            ts.append(tra())

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)
    x = fourmomenta_original.clone()
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
        [tr.EPPP_to_PPPM2],
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
    ],
)
@pytest.mark.parametrize(
    "distribution",
    [
        NaiveDistribution,
        NaiveLogDistribution,
        FourmomentaDistribution,
        JetmomentaDistribution,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [10000])
def test_jacobians(transforms, distribution, experiment_np, nevents):
    """test forward() and inverse() methods"""
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.base_kwargs,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    device = torch.device("cpu")
    dtype = torch.float64  # sometimes fails with torch32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_LogPt:
            ts.append(tra(exp.pt_min, exp.units))
        else:
            ts.append(tra())

    shape = (nevents, nparticles, 4)
    fourmomenta_original = d.sample(shape, device, dtype)
    x = fourmomenta_original.clone()
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
    grad_outputs = torch.ones_like(x, dtype=dtype)
    jac_fw_autograd = torch.autograd.grad(y, x, grad_outputs=grad_outputs)[0]
    jac_inv_autograd = torch.autograd.grad(z, y, grad_outputs=grad_outputs)[0]

    # multiply by unit vector (to be able to compare with autograd)
    jac_fw = jac_fw.sum(dim=-2)
    jac_inv = jac_inv.sum(dim=-2)

    """
    # alternative autograd implementation that computes the full jacobian matrix
    # this does not work yet...
    jac_fw_autograd, jac_inv_autograd = [], []
    for i in range(4):
        grad_outputs = torch.zeros_like(x)
        grad_outputs[...,i] = 1.
        fw_autograd = torch.autograd.grad(y, x, grad_outputs=grad_outputs,
                                          retain_graph=True, create_graph=False, allow_unused=True)[0]
        inv_autograd = torch.autograd.grad(z, y, grad_outputs=grad_outputs,
                                          retain_graph=True, create_graph=False, allow_unused=True)[0]
        jac_fw_autograd.append(fw_autograd)
        jac_inv_autograd.append(inv_autograd)
    jac_fw_autograd = torch.stack(jac_fw_autograd, dim=-2)
    jac_inv_autograd = torch.stack(jac_inv_autograd, dim=-2)
    """

    # compare jacobian to autograd
    torch.testing.assert_close(jac_fw, jac_fw_autograd, **TOLERANCES)
    torch.testing.assert_close(jac_inv, jac_inv_autograd, **TOLERANCES)
