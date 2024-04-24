import math
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from experiments.eventgen.wrappers import GATrCFM
from experiments.eventgen.coordinates import (
    convert_coordinates,
    convert_velocity,
)
from experiments.toptagging.dataset import embed_beam_reference
from gatr.interface import extract_scalar, embed_vector


class AfterBurner(GATrCFM):
    # Abstract afterburner class

    def __init__(self, afterburner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.cfm.coordinates_sampling == self.cfm.coordinates_loss
        assert self.cfm.coordinates_sampling == self.cfm.coordinates_straight
        assert self.cfm.coordinates_sampling != "Fourmomenta"
        self.afterburner = afterburner

    def afterburn(self, v_sampling, x_sampling, v_fourmomenta, x_fourmomenta):
        # default: do nothing
        return v_sampling

    def batch_loss(self, x0_fourmomenta, ijet):
        """
        Modified version of CFM.batch_loss
        """
        t = torch.rand(
            x0_fourmomenta.shape[0],
            1,
            1,
            dtype=x0_fourmomenta.dtype,
            device=x0_fourmomenta.device,
        )
        x1_fourmomenta = self.sample_base(
            x0_fourmomenta.shape, x0_fourmomenta.device, x0_fourmomenta.dtype
        )

        # construct trajectories in coordinates_straight
        x0_straight = self.coordinates_straight.fourmomenta_to_x(x0_fourmomenta)
        x1_straight = self.coordinates_straight.fourmomenta_to_x(x1_fourmomenta)
        xt_straight, vt_straight = self.coordinates_straight.get_trajectory(
            x0_straight, x1_straight, t
        )

        # predict velocity in coordinates_network
        xt_network = convert_coordinates(
            xt_straight, self.coordinates_straight, self.coordinates_network
        )
        vp_network = self.get_velocity(xt_network, t, ijet=ijet)

        # transform all velocities to coordinates_loss
        vp_loss = convert_velocity(
            vp_network, xt_network, self.coordinates_network, self.coordinates_loss
        )[0]
        vt_loss, xt_loss = convert_velocity(
            vt_straight, xt_straight, self.coordinates_straight, self.coordinates_loss
        )
        vt_loss = self.afterburn(vt_loss, xt_loss, vp_network, xt_network)  # NEW LINE

        loss = self.loss(vp_loss, vt_loss)
        return loss, [self.loss(vp_loss[..., i], vt_loss[..., i]) for i in range(4)]

    def sample(
        self, ijet, shape, device, dtype, trajectory_path=None, n_trajectories=100
    ):
        """
        Modified version of CFM.sample
        """
        # overhead for saving trajectories
        save_trajectory = trajectory_path is not None
        if save_trajectory:
            xts_sampling, vts_sampling, ts = [], [], []

        def velocity(t, xt_sampling):
            t = t * torch.ones(
                shape[0], 1, 1, dtype=xt_sampling.dtype, device=xt_sampling.device
            )
            xt_network = convert_coordinates(
                xt_sampling, self.coordinates_sampling, self.coordinates_network
            )
            vt_network = self.get_velocity(xt_network, t, ijet=ijet)
            vt_sampling, xt_sampling = convert_velocity(
                vt_network,
                xt_network,
                self.coordinates_network,
                self.coordinates_sampling,
            )
            vt_sampling = self.afterburn(
                vt_sampling, xt_sampling, vt_network, xt_network
            )  # NEW LINE

            # collect trajectories
            if save_trajectory:
                xts_sampling.append(xt_sampling[:n_trajectories, ...])
                vts_sampling.append(vt_sampling[:n_trajectories, ...])
                ts.append(t[0, 0, 0])
            return vt_sampling

        # sample fourmomenta from base distribution
        x1_fourmomenta = self.sample_base(shape, device, dtype)
        x1_sampling = self.coordinates_sampling.fourmomenta_to_x(x1_fourmomenta)

        # solve ODE in sampling space
        x0_sampling = odeint(
            velocity,
            x1_sampling,
            torch.tensor([1.0, 0.0]),
            **self.odeint,
        )[-1]

        # transform generated event back to fourmomenta
        x0_fourmomenta = self.coordinates_sampling.x_to_fourmomenta(x0_sampling)

        # save trajectories to file
        if save_trajectory:
            # collect trajectories
            xts_sampling = torch.stack(xts_sampling, dim=0)
            vts_sampling = torch.stack(vts_sampling, dim=0)
            ts = torch.stack(ts, dim=0)

            # determine true trajectories
            xts_straight = convert_coordinates(
                xts_sampling, self.coordinates_sampling, self.coordinates_straight
            )
            vts_straight_t, xts_straight_t = self.coordinates_straight.get_trajectory(
                xts_straight[-1, ...]
                .reshape(1, *xts_straight.shape[1:])
                .expand(xts_straight.shape),
                xts_straight[0, ...]
                .reshape(1, *xts_straight.shape[1:])
                .expand(xts_straight.shape),
                ts.reshape(ts.shape[0], 1, 1, 1),
            )

            # transform to fourmomenta space
            (
                vts_fourmomenta_t,
                xts_fourmomenta_t,
            ) = self.coordinates_straight.velocity_x_to_fourmomenta(
                vts_straight_t, xts_straight_t
            )
            (
                vts_fourmomenta,
                xts_fourmomenta,
            ) = self.coordinates_sampling.velocity_x_to_fourmomenta(
                vts_sampling, xts_sampling
            )

            # save
            np.savez_compressed(
                trajectory_path,
                xts_learned=xts_fourmomenta.cpu() * self.units,
                vts_learned=vts_fourmomenta.cpu() * self.units,
                xts_true=xts_fourmomenta_t.cpu() * self.units,
                vts_true=vts_fourmomenta_t.cpu() * self.units,
                ts=ts.cpu(),
            )

        return x0_fourmomenta

    def log_prob(self, x0_fourmomenta, ijet):
        """
        Modified version of CFM.log_prob
        """

        def net_wrapper(t, state):
            with torch.set_grad_enabled(True):
                xt_sampling = state[0].detach().requires_grad_(True)
                t = t * torch.ones(
                    xt_sampling.shape[0],
                    1,
                    1,
                    dtype=xt_sampling.dtype,
                    device=xt_sampling.device,
                )
                xt_network = convert_coordinates(
                    xt_sampling, self.coordinates_sampling, self.coordinates_network
                )
                vt_network = self.get_velocity(xt_network, t, ijet=ijet)
                vt_sampling, xt_sampling = convert_velocity(
                    vt_network,
                    xt_network,
                    self.coordinates_network,
                    self.coordinates_sampling,
                )
                vt_sampling = self.afterburn(
                    vt_sampling, xt_sampling, vt_network, xt_network
                )  # NEW LINE
                dlogp_dt_sampling = (
                    -self.trace_fn(vt_sampling, xt_sampling).unsqueeze(-1).detach()
                )  # note: trace affected by afterburner
            return vt_sampling.detach(), dlogp_dt_network

        # solve ODE in sampling space
        x0_sampling = self.coordinates_sampling.fourmomenta_to_x(x0_fourmomenta)
        logdetjac0_cfm_sampling = torch.zeros(
            (x0_sampling.shape[0], 1),
            dtype=x0_sampling.dtype,
            device=x0_sampling.device,
        )
        state0 = (x0_sampling, logdetjac0_cfm_sampling)
        xt_sampling, logdetjact_cfm_sampling = odeint(
            net_wrapper,
            state0,
            torch.tensor(
                [0.0, 1.0], dtype=x0_sampling.dtype, device=x0_sampling.device
            ),
            **self.odeint,
        )
        logdetjac_cfm_sampling = logdetjact_cfm_sampling[-1].detach()
        x1_sampling = xt_sampling[-1].detach()
        x1_fourmomenta = self.coordinates_sampling.x_to_fourmomenta(x1_sampling)
        logdetjac_forward = self.coordinates_sampling.logdetjac_fourmomenta_to_x(
            x0_fourmomenta
        )[0]
        logdetjac_inverse = -self.coordinates_sampling.logdetjac_fourmomenta_to_x(
            x1_fourmomenta
        )[0]

        # collect log_probs
        log_prob_base_fourmomenta = self.distribution.log_prob(x1_fourmomenta)
        log_prob_fourmomenta = (
            log_prob_base_fourmomenta
            - logdetjac_cfm_network
            - logdetjac_forward
            - logdetjac_inverse
        )
        return log_prob_fourmomenta


class Nuke(AfterBurner):
    def afterburn(self, v_sampling, x_sampling, v_fourmomenta, x_fourmomenta):
        x = torch.cat([v_sampling, x_sampling, v_fourmomenta, x_fourmomenta], dim=-1)
        v_modified = self.afterburner(x)
        v_sampling[..., [0, 3]] = v_modified
        return v_sampling


class EquiBurner(AfterBurner):
    def afterburn(self, v_sampling, x_sampling, v_fourmomenta, x_fourmomenta):
        # scalar embedding
        s = torch.cat([x_sampling, v_sampling], dim=-1)

        # mv embedding
        mv = torch.stack([x_fourmomenta, v_fourmomenta], dim=-2)
        mv = embed_vector(mv)
        beam = embed_beam_reference(
            mv, self.beam_reference, self.add_time_reference, self.two_beams
        )
        if beam is not None:
            beam = (
                beam.unsqueeze(0)
                .unsqueeze(0)
                .expand(*mv.shape[:-2], beam.shape[-2], 16)
            )
            mv = torch.cat([mv, beam], dim=-2)

        # call afterburner
        mv_outputs, s_outputs = self.afterburner(mv, s)

        # modify velocities
        v_modified = extract_scalar(mv_outputs).squeeze()
        v_sampling[..., [0, 3]] = v_modified
        return v_sampling
