import math
import numpy as np
import torch
from torch import nn
from torch.autograd import grad

from torchdiffeq import odeint
from experiments.eventgen.distributions import (
    BaseDistribution,
    NaivePPPM2,
    NaivePPPLogM2,
    StandardPPPLogM2,
    StandardLogPtPhiEtaLogM2,
)
import experiments.eventgen.coordinates as c
import experiments.eventgen.distributions as d
from experiments.eventgen.coordinates import (
    convert_coordinates,
    convert_velocity,
)


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, input_dim=1, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.weights = nn.Parameter(
            scale * torch.randn(input_dim, embed_dim // 2), requires_grad=False
        )

    def forward(self, t):
        projection = 2 * math.pi * torch.matmul(t, self.weights)
        embedding = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
        return embedding


def hutchinson_trace(x_out, x_in):
    # Hutchinson's trace Jacobian estimator, needs O(1) calls to autograd
    noise = torch.randint_like(x_in, low=0, high=2).float() * 2 - 1.0
    x_out_noise = torch.sum(x_out * noise)
    gradient = grad(x_out_noise, x_in)[0].detach()
    return torch.sum(gradient * noise, dim=[1, 2])


def autograd_trace(x_out, x_in):
    # Standard way of calculating trace of the Jacobian, needs O(n) calls to autograd
    trJ = 0.0
    for i in range(x_out.shape[1]):
        for j in range(x_out.shape[2]):
            trJ += (
                grad(x_out[:, i, j].sum(), x_in, retain_graph=True)[0]
                .contiguous()[:, i, j]
                .contiguous()
                .detach()
            )
    return trJ.contiguous()


class CFM(nn.Module):
    """
    Base class for all CFM models
    - event-generation-specific features are implemented in EventCFM
    - get_velocity is implemented by architecture-specific subclasses
    """

    def __init__(
        self,
        cfm,
        odeint={"method": "dopri5", "atol": 1e-5, "rtol": 1e-5, "options": None},
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=cfm.embed_t_dim, scale=cfm.embed_t_scale
            ),
            nn.Linear(cfm.embed_t_dim, cfm.embed_t_dim),
        )
        self.trace_fn = hutchinson_trace if cfm.hutchinson else autograd_trace
        self.odeint = odeint
        self.cfm = cfm
        self.loss = lambda v1, v2: nn.functional.mse_loss(v1, v2)

        # initialize to base objects, this will be overwritten later
        self.distribution = BaseDistribution()
        self.coordinates_straight = c.BaseCoordinates()
        self.coordinates_network = c.BaseCoordinates()
        self.coordinates_sampling = c.BaseCoordinates()
        self.coordinates_list = [
            self.coordinates_straight,
            self.coordinates_network,
            self.coordinates_sampling,
        ]

        if cfm.transforms_float64:
            c.DTYPE = torch.float64
        else:
            c.DTYPE = torch.float32

    def init_distribution(self):
        raise NotImplementedError

    def init_coordinates(self):
        raise NotImplementedError

    def sample_base(self, shape, device, dtype, generator=None):
        fourmomenta = self.distribution.sample(
            shape, device, dtype, generator=generator
        )
        return fourmomenta

    def get_velocity(self, x, t, ijet):
        """
        Parameters
        ----------
        x : torch.tensor with shape (batchsize, n_particles, 4)
        t : torch.tensor with shape (batchsize, 1, 1)
        ijet: int
        """
        # implemented by architecture-specific subclasses
        raise NotImplementedError

    def get_velocity_sampling(self, xt_network, t, ijet):
        """
        Parameters
        ----------
        xt_network : torch.tensor with shape (batchsize, n_particles, 4)
        t : torch.tensor with shape (batchsize, 1, 1)
        ijet: int
        """
        # This method is only overwritten by GATrCFM and GAPCFM
        vp_network = self.get_velocity(xt_network, t, ijet=ijet)
        vp_sampling, xt_sampling = convert_velocity(
            vp_network, xt_network, self.coordinates_network, self.coordinates_sampling
        )
        return vp_sampling, xt_sampling

    def handle_velocity(self, v):
        # default: do nothing
        return v

    def batch_loss(self, x0_fourmomenta, ijet):
        """
        Construct the conditional flow matching objective

        Parameters
        ----------
        x0_fourmomenta : torch.tensor with shape (batchsize, n_particles, 4)
            Target space particles in fourmomenta space
        ijet: int
            Process information (eg ttbar+0j vs ttbar+1j)
            Only used in transformer architectures, ignored for MLP and GAP

        Returns
        -------
        loss : torch.tensor with shape (1)
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

        # construct target trajectories
        x0_straight = self.coordinates_straight.fourmomenta_to_x(x0_fourmomenta)
        x1_straight = self.coordinates_straight.fourmomenta_to_x(x1_fourmomenta)
        xt_straight, vt_straight = self.coordinates_straight.get_trajectory(
            x0_straight, x1_straight, t
        )
        vt_sampling = convert_velocity(
            vt_straight,
            xt_straight,
            self.coordinates_straight,
            self.coordinates_sampling,
        )[0]

        # predict velocity
        xt_network = convert_coordinates(
            xt_straight, self.coordinates_straight, self.coordinates_network
        )
        vp_sampling = self.get_velocity_sampling(xt_network, t, ijet=ijet)[0]

        # evaluate conditional flow matching objective
        loss = self.loss(vp_sampling, vt_sampling)
        return loss, [
            self.loss(vp_sampling[..., i], vt_sampling[..., i]) for i in range(4)
        ]

    def sample(
        self, ijet, shape, device, dtype, trajectory_path=None, n_trajectories=100
    ):
        """
        Sample from CFM model
        Solve an ODE using a NN-parametrized velocity field

        Parameters
        ----------
        ijet : int
            Process information (eg ttbar+0j vs ttbar+1j)
            Only used in transformer architectures
        shape : List[int]
            Shape of events that should be generated
        device : torch.device
        dtype : torch.dtype
        trajectory_path : str
            path where trajectories should be saved
            no trajectories will be saved if 'trajectory_path is None'
        n_trajectories: int
            Number of trajectories to keep, out of the full batchsize trajectories

        Returns
        -------
        x0_fourmomenta : torch.tensor with shape shape = (batchsize, n_particles, 4)
            Generated events
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
            vt_sampling, xt_sampling = self.get_velocity_sampling(
                xt_network, t, ijet=ijet
            )
            vt_sampling = self.handle_velocity(vt_sampling)

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

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # and all components of the event are nan...
        # just sample another event in this case)
        mask = torch.isfinite(x0_sampling).all(dim=[1, 2])
        x0_sampling = x0_sampling[mask, ...]
        x1_fourmomenta = x1_fourmomenta[mask, ...]

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
        Evaluate log_prob for existing target samples in a CFM model
        Solve ODE involving the trace of the velocity field, this is more expensive than normal sampling
        The 'self.hutchinson' parameter controls if the trace should be evaluated
        with the hutchinson trace estimator that needs O(1) calls to the network,
        as opposed to the exact autograd trace that needs O(n_particles) calls to the network
        Note: Could also have a sample_and_log_prob method, but we have no use case for this

        Parameters
        ----------
        x0_fourmomenta : torch.tensor with shape (batchsize, n_particles, 4)
            Target space particles in fourmomenta space
        ijet: int
            Process information (eg ttbar+0j vs ttbar+1j)
            Only used in transformer architectures

        Returns
        -------
        log_prob_fourmomenta : torch.tensor with shape (batchsize)
            log_prob of each event in x0, evaluated in fourmomenta space
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
                vt_sampling = self.get_velocity_sampling(xt_network, t, ijet=ijet)[0]
                vt_sampling = self.handle_velocity(vt_sampling)
                dlogp_dt_sampling = (
                    -self.trace_fn(vt_sampling, xt_sampling).unsqueeze(-1).detach()
                )
            return vt_sampling, dlogp_dt_sampling

        # solve ODE in coordinates_sampling
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

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # just remove these events from the log_prob computation)
        mask = torch.isfinite(x1_sampling).all(dim=[1, 2])
        logdetjac_cfm_sampling = logdetjac_cfm_sampling[mask, ...]
        x1_sampling = x1_sampling[mask, ...]
        x0_fourmomenta = x0_fourmomenta[mask, ...]

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
            - logdetjac_cfm_sampling
            - logdetjac_forward
            - logdetjac_inverse
        )

        # the infamous clipper
        # (MLP sometimes has single large-NLL events -> exclude those from NLL computation)
        log_prob_fourmomenta = log_prob_fourmomenta[log_prob_fourmomenta > -100]
        return log_prob_fourmomenta


class EventCFM(CFM):
    """
    Add event-generation-specific methods to CFM classes:
    - Save information at the wrapper level
    - Handle base distribution and coordinates for RFM
    - Wrapper-specific preprocessing and undo_preprocessing
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_physics(
        self,
        units,
        pt_min,
        delta_r_min,
        onshell_list,
        onshell_mass,
        base_type,
        use_pt_min,
        use_delta_r_min,
    ):
        """
        Pass physics information to the CFM class

        Parameters
        ----------
        units: float
            Scale of dimensionful quantities
            I call it 'units' because we can really choose it arbitrarily without losing anything
            Hard-coded in EventGenerationExperiment
        pt_min: List[float]
            Minimum pt value for each particle
            Hard-coded in EventGenerationExperiment
        delta_r_min: float
            Minimum delta_r value
            We do not support different minimum delta_r for each particle yet
            Hard-coded in EventGenerationExperiment
        onshell_list: List[int]
            Indices of the onshell particles
            Hard-coded in EventGenerationExperiment
        onshell_mass: List[float]
            Masses of the onshell particles in the same order as in onshell_list
            Hard-coded in EventGenerationExperiment
        base_type: int
            Which base distribution to use
        use_delta_r_min: bool
            Whether the base distribution should have delta_r cuts
        use_pt_min: bool
            Whether the base distribution should have pt cuts
        """
        self.units = units
        self.pt_min = pt_min
        self.delta_r_min = delta_r_min
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.base_type = base_type
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min

        # same preprocessing for all multiplicities
        self.prep_params = {}

    def init_distribution(self):
        args = [
            self.onshell_list,
            self.onshell_mass,
            self.units,
            self.delta_r_min,
            self.pt_min,
            self.use_delta_r_min,
            self.use_pt_min,
        ]
        if self.base_type == 1:
            self.distribution = NaivePPPM2(*args)
        elif self.base_type == 2:
            self.distribution = NaivePPPLogM2(*args)
        elif self.base_type == 3:
            self.distribution = StandardPPPLogM2(*args)
        elif self.base_type == 4:
            self.distribution = StandardLogPtPhiEtaLogM2(*args)
        else:
            raise ValueError(f"base_type={self.base_type} not implemented")

    def init_coordinates(self):
        self.coordinates_network = self._init_coordinates(self.cfm.coordinates_network)
        self.coordinates_straight = self._init_coordinates(
            self.cfm.coordinates_straight
        )
        self.coordinates_sampling = self._init_coordinates(
            self.cfm.coordinates_sampling
        )
        self.coordinates = [
            self.coordinates_straight,
            self.coordinates_network,
            self.coordinates_sampling,
        ]

    def _init_coordinates(self, coordinates_label):
        if coordinates_label == "Fourmomenta":
            coordinates = c.Fourmomenta
        elif coordinates_label == "M2PPP":
            coordinates = c.M2PPP()
        elif coordinates_label == "PPPM2":
            coordinates = c.PPPM2()
        elif coordinates_label == "PPPLogM2":
            coordinates = c.PPPLogM2()
        elif coordinates_label == "StandardPPPLogM2":
            coordinates = c.StandardPPPLogM2()
        elif coordinates_label == "EPhiPtPz":
            coordinates = c.EPhiPtPz()
        elif coordinates_label == "PtPhiEtaE":
            coordinates = c.PtPhiEtaE()
        elif coordinates_label == "PtPhiEtaM2":
            coordinates = c.PtPhiEtaM2()
        elif coordinates_label == "LogPtPhiEtaE":
            coordinates = c.LogPtPhiEtaE(self.pt_min, self.units)
        elif coordinates_label == "LogPtPhiEtaM2":
            coordinates = c.LogPtPhiEtaM2(self.pt_min, self.units)
        elif coordinates_label == "PtPhiEtaLogM2":
            coordinates = c.PtPhiEtaLogM2()
        elif coordinates_label == "LogPtPhiEtaLogM2":
            coordinates = c.LogPtPhiEtaLogM2(self.pt_min, self.units)
        elif coordinates_label == "StandardLogPtPhiEtaLogM2":
            coordinates = c.StandardLogPtPhiEtaLogM2(self.pt_min, self.units)
        elif coordinates_label == "StandardLogPtPhiEtaLogM2_from_M2PPP":
            assert hasattr(self, "coordinates_network")
            assert isinstance(self.coordinates_network, c.M2PPP), (
                f"Trying to use coordinates=StandardLogPtPhiEtaLogM2_from_M2PPP, "
                f"but coordinates=StandardLogPtPhiEtaLogM2_from_M2PPP can only be used with coordinates_network=M2PPP"
            )
            coordinates = c.StandardLogPtPhiEtaLogM2_from_M2PPP(
                self.pt_min, self.units, self.onshell_list
            )
        else:
            raise ValueError(f"coordinates={coordinates_label} not implemented")
        return coordinates

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        return fourmomenta

    def undo_preprocess(self, fourmomenta):
        fourmomenta = fourmomenta * self.units
        return fourmomenta

    def sample(self, *args, **kwargs):
        fourmomenta = super().sample(*args, **kwargs)
        return fourmomenta

    def handle_velocity(self, v):
        if self.cfm.coordinates_straight in [
            "PPPM2",
            "PPPLogM2",
            "StandardPPPLogM2",
            "PtPhiEtaM2",
            "LogPtPhiEtaM2",
            "PtPhiEtaLogM2",
            "LogPtPhiEtaLogM2",
            "StandardLogPtPhiEtaLogM2",
        ]:
            # manually set mass velocity of onshell events to zero
            v[..., self.onshell_list, 3] = 0.0
        return v
