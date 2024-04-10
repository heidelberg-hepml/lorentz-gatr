import pytest
import torch
import numpy as np

import experiments.eventgen.transforms as tr
from experiments.eventgen.distributions import NaiveLogDistribution
from tests.helpers import MILD_TOLERANCES as TOLERANCES


def test_simple():
    """Some very simple tests"""
    fourmomentum = torch.tensor([[1, 1, 0, 0], [1, 1, 0, -1]]).float()
    ptphietam2 = torch.tensor(
        [[1, 0, 0, 0], [1, 0, np.arctanh(-1 / 2**0.5), -1]]
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
        [tr.EPPP_to_PPPM2, tr.M2_to_logM2],
        [tr.EPPP_to_PtPhiEtaE, tr.Pt_to_logPt],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.M2_to_logM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.Pt_to_logPt],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_logM2,
            tr.Pt_to_logPt,
        ],
    ],
)
@pytest.mark.parametrize("nevents", [100000])
@pytest.mark.parametrize("nparticles", [10])
def test_invertibility(transforms, nevents, nparticles):
    """test forward() and inverse() methods"""
    args = [[], [], None, None, None, [0.0] * nparticles, False, False]
    d = NaiveLogDistribution(*args)
    device = torch.device("cpu")
    dtype = torch.float32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_logPt:
            ts.append(tra(args[5]))
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
        [tr.EPPP_to_PPPM2, tr.M2_to_logM2],
        [tr.EPPP_to_PtPhiEtaE, tr.Pt_to_logPt],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.M2_to_logM2],
        [tr.EPPP_to_PtPhiEtaE, tr.PtPhiEtaE_to_PtPhiEtaM2, tr.Pt_to_logPt],
        [
            tr.EPPP_to_PtPhiEtaE,
            tr.PtPhiEtaE_to_PtPhiEtaM2,
            tr.M2_to_logM2,
            tr.Pt_to_logPt,
        ],
    ],
)
@pytest.mark.parametrize("nevents", [100000])
@pytest.mark.parametrize("nparticles", [10])
def test_jacobians(transforms, nevents, nparticles):
    """
    test the _jac_forward and _jac_inverse methods with a call to autograd
    only the last transform in transforms is tested
    """
    args = [[], [], None, None, None, [0.0] * nparticles, False, False]
    d = NaiveLogDistribution(*args)
    device = torch.device("cpu")
    dtype = torch.float32
    ts = []
    for tra in transforms:
        if tra == tr.Pt_to_logPt:
            ts.append(tra(args[5]))
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
    diag = torch.eye(4)[None,None,...].expand(diag_left.shape)
    torch.testing.assert_close(diag_right, diag, **TOLERANCES)
    torch.testing.assert_close(diag_left, diag, **TOLERANCES)

    # jacobians from autograd
    grad_outputs = torch.ones_like(x)
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
