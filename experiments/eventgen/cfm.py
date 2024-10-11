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
from experiments.eventgen.utils import GaussianFourierProjection
import experiments.eventgen.coordinates as c
from experiments.eventgen.geometry import BaseGeometry, SimplePossiblyPeriodicGeometry
from experiments.eventgen.mfm import MassMFM, LANDMFM


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

        # initialize to base objects, this will be overwritten later
        self.distribution = BaseDistribution()
        self.coordinates = c.BaseCoordinates()
        self.geometry = BaseGeometry()

        if cfm.transforms_float64:
            c.DTYPE = torch.float64
        else:
            c.DTYPE = torch.float32

    def init_distribution(self):
        raise NotImplementedError

    def init_coordinates(self):
        raise NotImplementedError

    def init_geometry(self):
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
        x0_straight = self.coordinates.fourmomenta_to_x(x0_fourmomenta)
        x1_straight = self.coordinates.fourmomenta_to_x(x1_fourmomenta)
        xt_straight, vt_straight = self.geometry.get_trajectory(
            x0_straight, x1_straight, t
        )
        vp_straight = self.get_velocity(xt_straight, t, ijet=ijet)

        # evaluate conditional flow matching objective
        distance = self.geometry.get_metric(
            vp_straight, vt_straight, xt_straight
        ).mean()
        distance_particlewise = [
            ((vp_straight - vt_straight) ** 2)[:, i].mean() for i in range(4)
        ]
        return distance, distance_particlewise

    def sample(self, ijet, shape, device, dtype):
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

        Returns
        -------
        x0_fourmomenta : torch.tensor with shape shape = (batchsize, n_particles, 4)
            Generated events
        """

        def velocity(t, xt_straight):
            t = t * torch.ones(
                shape[0], 1, 1, dtype=xt_straight.dtype, device=xt_straight.device
            )
            vt_straight = self.get_velocity(xt_straight, t, ijet=ijet)
            vt_straight = self.handle_velocity(vt_straight)
            return vt_straight

        # sample fourmomenta from base distribution
        x1_fourmomenta = self.sample_base(shape, device, dtype)
        x1_straight = self.coordinates.fourmomenta_to_x(x1_fourmomenta)

        # solve ODE in straight space
        x0_straight = odeint(
            velocity,
            x1_straight,
            torch.tensor([1.0, 0.0]),
            **self.odeint,
        )[-1]

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # and all components of the event are nan...
        # just sample another event in this case)
        mask = torch.isfinite(x0_straight).all(dim=[1, 2])
        x0_straight = x0_straight[mask, ...]
        x1_fourmomenta = x1_fourmomenta[mask, ...]

        # transform generated event back to fourmomenta
        x0_fourmomenta = self.coordinates.x_to_fourmomenta(x0_straight)
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
                xt_straight = state[0].detach().requires_grad_(True)
                t = t * torch.ones(
                    xt_straight.shape[0],
                    1,
                    1,
                    dtype=xt_straight.dtype,
                    device=xt_straight.device,
                )
                vt_straight = self.get_velocity(xt_straight, t, ijet=ijet)
                vt_straight = self.handle_velocity(vt_straight)
                dlogp_dt_straight = (
                    -self.trace_fn(vt_straight, xt_straight).unsqueeze(-1).detach()
                )
            return vt_straight, dlogp_dt_straight

        # solve ODE in coordinates_straight
        x0_straight = self.coordinates.fourmomenta_to_x(x0_fourmomenta)
        logdetjac0_cfm_straight = torch.zeros(
            (x0_straight.shape[0], 1),
            dtype=x0_straight.dtype,
            device=x0_straight.device,
        )
        state0 = (x0_straight, logdetjac0_cfm_straight)
        xt_straight, logdetjact_cfm_straight = odeint(
            net_wrapper,
            state0,
            torch.tensor(
                [0.0, 1.0], dtype=x0_straight.dtype, device=x0_straight.device
            ),
            **self.odeint,
        )
        logdetjac_cfm_straight = logdetjact_cfm_straight[-1].detach()
        x1_straight = xt_straight[-1].detach()

        # the infamous nan remover
        # (MLP sometimes returns nan for single events,
        # just remove these events from the log_prob computation)
        mask = torch.isfinite(x1_straight).all(dim=[1, 2])
        logdetjac_cfm_straight = logdetjac_cfm_straight[mask, ...]
        x1_straight = x1_straight[mask, ...]
        x0_fourmomenta = x0_fourmomenta[mask, ...]

        x1_fourmomenta = self.coordinates.x_to_fourmomenta(x1_straight)
        logdetjac_forward = self.coordinates.logdetjac_fourmomenta_to_x(x0_fourmomenta)[
            0
        ]
        logdetjac_inverse = -self.coordinates.logdetjac_fourmomenta_to_x(
            x1_fourmomenta
        )[0]

        # collect log_probs
        log_prob_base_fourmomenta = self.distribution.log_prob(x1_fourmomenta)
        log_prob_fourmomenta = (
            log_prob_base_fourmomenta
            - logdetjac_cfm_straight
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
        virtual_components,
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
        virtual_components: List[List[int]]
            Indices of the virtual particles
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
        self.virtual_components = virtual_components
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
        self.coordinates = self._init_coordinates(self.cfm.coordinates_straight)

    def _init_coordinates(self, coordinates_label):
        if coordinates_label == "Fourmomenta":
            coordinates = c.Fourmomenta()
        elif coordinates_label == "PPPM2":
            coordinates = c.PPPM2()
        elif coordinates_label == "PPPLogM2":
            coordinates = c.PPPLogM2()
        elif coordinates_label == "StandardPPPLogM2":
            coordinates = c.StandardPPPLogM2(self.onshell_list)
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
            coordinates = c.StandardLogPtPhiEtaLogM2(
                self.pt_min,
                self.units,
            )
        else:
            raise ValueError(f"coordinates={coordinates_label} not implemented")
        return coordinates

    def init_geometry(self, fourmomenta, **kwargs):
        # placeholder for any initialization that needs to be done
        if self.cfm.geometry.type == "simple":
            self.geometry = SimplePossiblyPeriodicGeometry(
                contains_phi=self.coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
            )
        elif self.cfm.geometry.type == "MassMFM":
            self.geometry = MassMFM(
                virtual_components=self.virtual_components,
                cfm=self.cfm,
                coordinates=self.coordinates,
                contains_phi=self.coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
            )
        elif self.cfm.geometry.type == "LANDMFM":
            self.geometry = LANDMFM(
                virtual_components=self.virtual_components,
                cfm=self.cfm,
                coordinates=self.coordinates,
                contains_phi=self.coordinates.contains_phi,
                periodic=self.cfm.geometry.periodic,
            )
        else:
            raise ValueError(f"geometry={self.cfm.geometry} not implemented")

        if self.cfm.geometry.type in ["MassMFM", "LANDMFM"]:
            assert (
                len(fourmomenta) == 1
            ), "MFM only implemented for single-multiplicity training for now"
            fourmomenta = fourmomenta[0]
            generator = torch.Generator().manual_seed(self.cfm.mfm.seed_base)
            base = self.sample_base(
                fourmomenta.shape,
                fourmomenta.device,
                fourmomenta.dtype,
                generator=generator,
            )
            self.geometry.initialize(base, fourmomenta, **kwargs)

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
        if self.coordinates.contains_mass:
            # manually set mass velocity of onshell events to zero
            v[..., self.onshell_list, 3] = 0.0
        return v
