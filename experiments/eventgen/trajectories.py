import math
import torch
import torchode as to

import experiments.eventgen.coordinates as c

ALPHA = 0.0

# maybe important from somewhere else?
def ensure_angle(x):
    # return (x+math.pi)%(2*math.pi) - math.pi
    return x


std_logmW2, mean_logmW2 = 2.2291, 1.4106
approx_prepmW2 = True


def get_prepd_mW2(x, c_start):
    # print(x)
    assert torch.isfinite(x).all()
    c_target = c.PtPhiEtaM2()
    jetmomenta = c.convert_coordinates(x, c_start, c_target)
    unpack = lambda x, i: [x[..., i, j] for j in range(4)]
    pt1, phi1, eta1, m21 = unpack(jetmomenta, 0)
    pt2, phi2, eta2, m22 = unpack(jetmomenta, 1)
    assert (pt1 > 0).all()
    assert (pt2 > 0).all()
    assert (m21 > 0).all()
    assert (m22 > 0).all()
    assert (phi1 > -math.pi).all() and (phi1 < math.pi).all()
    assert (phi2 > -math.pi).all() and (phi2 < math.pi).all()
    if approx_prepmW2:
        t1 = torch.cosh(eta1) * torch.cosh(eta2)
    else:
        t1_a = (m21 / pt1**2 + torch.cosh(eta1) ** 2) ** 0.5
        t1_b = (m22 / pt2**2 + torch.cosh(eta2) ** 2) ** 0.5
        t1 = t1_a * t1_b
    t2 = -torch.sinh(eta1) * torch.sinh(eta2)
    t3 = -torch.sin(phi1) * torch.sin(phi2)
    t4 = -torch.cos(phi1) * torch.cos(phi2)
    bracket = t1 + t2 + t3 + t4
    assert (bracket > 0).all()
    if approx_prepmW2:
        mW2 = 2 * pt1 * pt2 * bracket
    else:
        mW2 = m21 + m22 + 2 * pt1 * pt2 * bracket
    assert torch.isfinite(mW2).all()
    assert (mW2 >= 0).all()
    # mW2 = mW2.log()
    mW2 = (mW2 - mean_logmW2) / std_logmW2
    return mW2


class PremetricVelocity:
    def _get_premetric(self, x1, x2):
        raise NotImplementedError

    def _get_gradpremetric(self, x1, x2):
        raise NotImplementedError

    def get_gradpremetric(self, x1, x2, **kwargs):
        gradpremetric = self._get_gradpremetric(x1, x2, **kwargs)
        gradpremetric[..., 1] = ensure_angle(gradpremetric[..., 1])
        return gradpremetric

    def get_velocity(self, xt, xstart, xtarget, t, eps=0.0, **kwargs):
        gradpremetric = self._get_gradpremetric(xt, xtarget, **kwargs)
        premetric = self._get_premetric(xstart, xtarget, **kwargs)
        norm2 = torch.sum(gradpremetric**2, dim=[-1, -2], keepdims=True)
        velocity = -premetric * gradpremetric / (norm2 + eps)
        velocity[..., 1] = ensure_angle(velocity[..., 1])
        return velocity


class MassVelocity(PremetricVelocity):
    def _get_premetric(self, x1, x2, c_start):
        diff_naive = x1 - x2
        diff_naive[..., 1] = ensure_angle(diff_naive[..., 1])
        premetric_naive = diff_naive**2 / 2
        premetric_naive = premetric_naive.sum(dim=[-1, -2])
        diff_mW2 = get_prepd_mW2(x1, c_start) - get_prepd_mW2(x2, c_start)
        premetric_mW2 = ALPHA * diff_mW2**2 / 2
        premetric = premetric_naive + premetric_mW2
        assert (premetric >= 0).all()
        return premetric[..., None, None]  # .sqrt()

    def _get_gradpremetric(self, x1, x2, **kwargs):
        # take jacobian from autograd for now
        x1.requires_grad_()
        premetric = self._get_premetric(x1, x2, **kwargs)
        grad_outputs = torch.ones_like(premetric)
        grad = torch.autograd.grad(premetric, x1, grad_outputs=grad_outputs)[0].detach()
        grad[..., 1] = ensure_angle(grad[..., 1])
        return grad


class SophisticatedMass(c.StandardLogPtPhiEtaLogM2):
    def get_trajectory(self, x1, x2, t):
        dtype = torch.float64
        x1, x2, t = x1.to(dtype), x2.to(dtype), t.to(dtype)
        unsqueeze = lambda x: x.view(*list(x.shape[:-1]), x.shape[-1] // 4, 4)
        squeeze = lambda x: x.reshape(*list(x.shape[:-2]), x.shape[-2] * x.shape[-1])

        xstart, xtarget = x1, x2
        # xstart, xtarget = x2, x1

        vel = MassVelocity()

        def velocity_fn(t, xt):
            # print("New iteration")
            # assert torch.isfinite(xt).all()
            xt = unsqueeze(xt).detach()
            xt = torch.where(torch.isnan(xt), torch.randn_like(xt), xt)  # TBD: fix this
            xt[..., 1] = ensure_angle(xt[..., 1])
            vt = vel.get_velocity(xt, xstart, xtarget, t, c_start=self)
            assert torch.isfinite(vt).all()
            # print(xt, vt)
            # print(vt.min(dim=0), vt.max(dim=0))
            return squeeze(vt)

        max_steps = 100
        t = squeeze(t)
        kwargs = {"rtol": 1e-5, "atol": 1e-5}

        term = to.ODETerm(velocity_fn)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(term=term, **kwargs)
        solver = to.AutoDiffAdjoint(
            step_method,
            step_size_controller,
            max_steps=max_steps,
            backprop_through_step_size_control=False,
        )

        sol = torch.compile(solver).solve(
            to.InitialValueProblem(y0=squeeze(xstart), t_eval=t)
        )
        xt = unsqueeze(sol.ys[:, 0, :])

        vt = unsqueeze(velocity_fn(t, squeeze(xt)))
        return xt.to(torch.float32), vt.to(torch.float32)
