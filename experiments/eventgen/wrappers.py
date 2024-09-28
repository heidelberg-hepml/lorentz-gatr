import torch
import numpy as np
from torch import nn

from gatr.interface import embed_vector, extract_vector
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.toptagging.dataset import embed_beam_reference
from experiments.eventgen.cfm import EventCFM

from experiments.eventgen.coordinates import (
    convert_velocity,
)


def get_type_token(x_ref, type_token_channels):
    # embed type_token
    type_token_raw = torch.arange(x_ref.shape[1], device=x_ref.device, dtype=torch.long)
    type_token = nn.functional.one_hot(type_token_raw, num_classes=type_token_channels)
    type_token = type_token.unsqueeze(0).expand(
        x_ref.shape[0], x_ref.shape[1], type_token_channels
    )
    return type_token


def get_process_token(x_ref, ijet, process_token_channels):
    # embed process_token
    process_token_raw = torch.tensor([ijet], device=x_ref.device, dtype=torch.long)
    process_token = nn.functional.one_hot(
        process_token_raw, num_classes=process_token_channels
    ).squeeze()
    process_token = process_token.unsqueeze(0).expand(
        x_ref.shape[1], process_token_channels
    )
    process_token = process_token.unsqueeze(0).expand(
        x_ref.shape[0], x_ref.shape[1], process_token_channels
    )
    return process_token


class MLPCFM(EventCFM):
    """
    Baseline MLP velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net

    def get_velocity(self, x, t, ijet):
        t_embedding = self.t_embedding(t).squeeze()
        x = x.reshape(x.shape[0], -1)

        x = torch.cat([x, t_embedding], dim=-1)
        v = self.net(x)
        v = v.reshape(v.shape[0], v.shape[1] // 4, 4)
        return v


class GAPCFM(EventCFM):
    """
    Baseline GAP velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        beam_reference,
        two_beams,
        add_time_reference,
        scalar_dims,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint=odeint,
        )
        self.net = net
        self.beam_reference = beam_reference
        self.two_beams = two_beams
        self.add_time_reference = add_time_reference
        self.scalar_dims = scalar_dims
        assert (
            self.cfm.coordinates_network == "Fourmomenta"
        ), f"GA-networks require coordinates_network=Fourmomenta"

    def get_velocity(self, fourmomenta, t, ijet):
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta, v_s

    def get_velocity_sampling(self, xt_network, t, ijet):
        # Predict velocities as usual
        vp_network, vp_scalar = self.get_velocity(xt_network, t, ijet=ijet)
        vp_sampling, xt_sampling = convert_velocity(
            vp_network, xt_network, self.coordinates_network, self.coordinates_sampling
        )

        # Overwrite transformed velocities with scalar outputs of GATr
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        vp_sampling[..., self.scalar_dims] = vp_scalar[..., self.scalar_dims]
        return vp_sampling, xt_sampling

    def embed_into_ga(self, x, t, ijet):
        # note: ijet is not used
        # (joint training only supported for transformers)

        # scalar embedding
        s = self.t_embedding(t).squeeze()

        # mv embedding
        mv = embed_vector(x.reshape(x.shape[0], -1, 4))
        beam = embed_beam_reference(
            mv, self.beam_reference, self.add_time_reference, self.two_beams
        )
        if beam is not None:
            beam = beam.unsqueeze(0).expand(*mv.shape[:-2], beam.shape[-2], 16)
            mv = torch.cat([mv, beam], dim=-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        s = s.reshape(*s.shape[:-1], s.shape[-1] // 4, 4)
        return v, s


class TransformerCFM(EventCFM):
    """
    Baseline Transformer velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        type_token_channels,
        process_token_channels,
        odeint,
    ):
        # See GATrCFM.__init__ for documentation
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels

    def get_velocity(self, x, t, ijet):
        # note: flow matching happens directly in x space
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)

        x = torch.cat([x, type_token, process_token, t_embedding], dim=-1)
        v = self.net(x)
        return v


class GATrCFM(EventCFM):
    """
    GATr velocity network
    """

    def __init__(
        self,
        net,
        cfm,
        type_token_channels,
        process_token_channels,
        beam_reference,
        two_beams,
        add_time_reference,
        scalar_dims,
        odeint,
    ):
        """
        Parameters
        ----------
        net : torch.nn.Module
        cfm : Dict
            Information about how to set up CFM
            technical keys: embed_t_dim, embed_t_scale, hutchinson, transforms_float64, eps1_pt, eps1_m2
            conceptional keys: coordinates_straight, coordinates_network, coordinates_sampling
        type_token_channels : int
            Number of different particle id's
            Used for one-hot encoding to break permutation symmetry
        process_token_channels : int
            Number of different process id's
            Used for one-hot encoding to break permutation symmetry
        beam_reference : str
            Type of beam reference used to break the Lorentz symmetry
            Options: [None, "xyplane", "spacelike", "lightlike", "timelike"]
            See experiments.toptagging.dataset.py::embed_beam_reference for details
        two_beams : bool
            If beam_reference in ["spacelike", "lightlike", "timelike"],
            decide whether only (alpha,0,0,1) or both (alpha,0,0,+/-1) are included
        add_time_reference : bool
            Whether time direction (1,0,0,0) is included to break Lorentz group down to SO(3)
            This is formally required, because equivariant generation on non-compact groups is not possible
        scalar_dims : List[int]
            Components within the used parametrization
            for which the equivariantly predicted velocity (using multivector channels)
            is overwritten by a scalar network output (using scalar channels)
            This is required whenever coordinates_network != coordinates_sampling,
            and the transformation between the two contains e.g. log transforms
        odeint : Dict
            ODE solver settings to be passed to torchdiffeq.odeint
        """
        super().__init__(
            cfm,
            odeint,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.beam_reference = beam_reference
        self.two_beams = two_beams
        self.add_time_reference = add_time_reference
        self.scalar_dims = scalar_dims
        assert (np.array(scalar_dims) < 4).all() and (np.array(scalar_dims) >= 0).all()
        assert (
            self.cfm.coordinates_network == "Fourmomenta"
        ), f"GA-networks require coordinates_network=Fourmomenta"

    def get_velocity(self, fourmomenta, t, ijet):
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta, v_s

    def get_velocity_sampling(self, xt_network, t, ijet):
        # Predict velocities as usual
        vp_network, vp_scalar = self.get_velocity(xt_network, t, ijet=ijet)
        vp_sampling, xt_sampling = convert_velocity(
            vp_network, xt_network, self.coordinates_network, self.coordinates_sampling
        )

        # Overwrite transformed velocities with scalar outputs of GATr
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        vp_sampling[..., self.scalar_dims] = vp_scalar[..., self.scalar_dims]
        return vp_sampling, xt_sampling

    def embed_into_ga(self, x, t, ijet):
        # scalar embedding
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        s = torch.cat([type_token, process_token, t_embedding], dim=-1)

        # mv embedding
        mv = embed_vector(x).unsqueeze(-2)
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

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v, s
