import math
import torch
import torchode as to

import experiments.eventgen.coordinates as c
import matplotlib.pyplot as plt

ALPHA = 0.0

# maybe important from somewhere else?
def ensure_angle(x):
    return (x + math.pi) % (2 * math.pi) - math.pi
    return x


std_logmW2, mean_logmW2 = 1.0, -1.5


def get_prepd_mW2(x, c_start, approx_mW2=True, use_logmW2=True):
    # c_target = c.PtPhiEtaM2()
    # x = c.convert_coordinates(x, c_start, c_target)
    assert isinstance(c_start, c.StandardLogPtPhiEtaLogM2)
    for transform in reversed(c_start.transforms[2:]):
        x = transform.inverse(x)
    unpack = lambda x, i: [x[..., i, j] for j in range(4)]
    pt1, phi1, eta1, m21 = unpack(x, 1)
    pt2, phi2, eta2, m22 = unpack(x, 2)
    if approx_mW2:
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
    if approx_mW2:
        mW2 = 2 * pt1 * pt2 * bracket
    else:
        mW2 = m21 + m22 + 2 * pt1 * pt2 * bracket
    logmW2 = mW2.log() if use_logmW2 else mW2
    logmW2 = (logmW2 - mean_logmW2) / std_logmW2
    assert torch.isfinite(logmW2).all()
    return logmW2


class PremetricVelocity:
    def _get_premetric(self, x1, x2):
        raise NotImplementedError

    def get_gradpremetric(self, x1, x2, **kwargs):
        gradpremetric = self._get_gradpremetric(x1, x2, **kwargs)
        gradpremetric[..., 1] = ensure_angle(gradpremetric[..., 1])
        return gradpremetric

    def _get_gradpremetric(self, x1, x2, **kwargs):
        # use autograd jacobian for now
        x1.requires_grad_()
        premetric = self._get_premetric(x1, x2, **kwargs)
        grad_outputs = torch.ones_like(premetric)
        grad = torch.autograd.grad(premetric, x1, grad_outputs=grad_outputs)[0].detach()
        return grad

    def get_velocity(self, xt, xstart, xtarget, t, eps=0.0, **kwargs):
        gradpremetric = self._get_gradpremetric(xt, xtarget, **kwargs)
        premetric = self._get_premetric(xstart, xtarget, **kwargs)
        norm2 = torch.sum(gradpremetric**2, dim=[-1, -2], keepdims=True)
        velocity = -premetric * gradpremetric / (norm2 + eps)
        velocity[..., 1] = ensure_angle(velocity[..., 1])
        return velocity


class StraightVelocity(PremetricVelocity):
    def _get_premetric(self, x1, x2, **kwargs):
        diff = x1 - x2
        diff[..., 1] = ensure_angle(diff[..., 1])
        premetric = diff**2 / 2
        premetric = premetric.sum(dim=[-1, -2], keepdims=True)
        # could do .sqrt() here, but have to adapt gradpremetric then (=ugly)
        return premetric

    def _get_gradpremetric(self, x1, x2, **kwargs):
        grad = x1 - x2
        return grad


class MassVelocity(PremetricVelocity):
    def _get_premetric(self, x1, x2, **kwargs):
        diff_naive = x1 - x2
        diff_naive[..., 1] = ensure_angle(diff_naive[..., 1])
        premetric_naive = diff_naive**2 / 2
        premetric_naive = premetric_naive.sum(dim=[-1, -2])
        diff_mW2 = get_prepd_mW2(x1, **kwargs) - get_prepd_mW2(x2, **kwargs)
        premetric_mW2 = ALPHA * diff_mW2**2 / 2
        premetric = premetric_naive + premetric_mW2
        assert (premetric >= 0).all()
        # premetric = premetric.sqrt()
        return premetric[..., None, None]


class SophisticatedMass(c.StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, **kwargs):
        super().__init__(**kwargs)
        self.cfm = cfm

    def get_trajectory(self, x1, x2, t):
        dtype = torch.float64 if self.cfm.trajs.use_float64 else torch.float32
        x1, x2, t = x1.to(dtype), x2.to(dtype), t.to(dtype)
        unsqueeze = lambda x: x.view(*list(x.shape[:-1]), x.shape[-1] // 4, 4)
        squeeze = lambda x: x.reshape(*list(x.shape[:-2]), x.shape[-2] * x.shape[-1])

        xstart, xtarget = (x1, x2) if self.cfm.trajs.physics_to_latent else (x2, x1)
        vel = StraightVelocity() if self.cfm.trajs.naive else MassVelocity()
        
        t = squeeze(t)
        t = torch.ones_like(t)
        if self.cfm.trajs.bootstrap_factor > 1:
            factor = self.cfm.trajs.bootstrap_factor
            batchsize_eff = xstart.shape[0] // factor
            xstart, xtarget = xstart[:batchsize_eff,...], xtarget[:batchsize_eff, ...]
            t0 = torch.zeros_like(t[:batchsize_eff, ...])
            t_eval = t.reshape(batchsize_eff, factor)
            t_eval = torch.cat((t0, t_eval), dim=-1)
        else:
            t_eval = torch.cat((torch.zeros_like(t), t), dim=-1)

        def velocity_fn(t, xt, xstart=xstart, xtarget=xtarget):
            xt = unsqueeze(xt).detach()
            xt[..., 1] = ensure_angle(xt[..., 1])
            vt = vel.get_velocity(
                xt,
                xstart,
                xtarget,
                t,
                c_start=self,
                approx_mW2=self.cfm.trajs.approx_mW2,
                use_logmW2=self.cfm.trajs.use_logmW2,
            )
            return squeeze(vt)

        max_steps = self.cfm.trajs.max_steps
        kwargs = {"rtol": self.cfm.trajs.rtol, "atol": self.cfm.trajs.atol}

        term = to.ODETerm(velocity_fn)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(term=term, **kwargs)
        solver = to.AutoDiffAdjoint(
            step_method,
            step_size_controller,
            max_steps=max_steps,
            backprop_through_step_size_control=False,
        )
        problem = to.InitialValueProblem(y0=squeeze(xstart), t_eval=t_eval)

        sol = torch.compile(solver).solve(problem)
        # print(sol.stats["n_f_evals"][0].item())
        if self.cfm.trajs.bootstrap_factor > 1:
            factor = self.cfm.trajs.bootstrap_factor
            xt = sol.ys[:, :-1, :].reshape(x1.shape)
            vt = unsqueeze(velocity_fn(squeeze(t), squeeze(xt),
                                       xstart=xstart.repeat(factor, 1, 1),
                                       xtarget=xtarget.repeat(factor, 1, 1)))
        else:
            xt = sol.ys[:, -1, :]
            xt = unsqueeze(xt)
            vt = unsqueeze(velocity_fn(squeeze(t), squeeze(xt)))

        return xt.to(torch.float32), vt.to(torch.float32)
