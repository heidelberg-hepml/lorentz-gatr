import torch
import numpy as np

from gatr.interface import embed_vector, extract_vector, embed_spurions
from experiments.eventgen.cfm import EventCFM
from experiments.eventgen.utils import get_type_token, get_process_token


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


class EventCFMForGA(EventCFM):
    def __init__(self, scalar_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scalar_dims = scalar_dims
        assert (np.array(scalar_dims) < 4).all() and (np.array(scalar_dims) >= 0).all()

    def get_velocity(self, x_straight, t, ijet):
        assert self.coordinates is not None
        x_fourmomenta = self.coordinates.x_to_fourmomenta(x_straight)

        mv, s = self.embed_into_ga(x_fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta, v_s = self.extract_from_ga(mv_outputs, s_outputs)

        v_straight = self.coordinates.velocity_fourmomenta_to_x(
            v_fourmomenta,
            x_fourmomenta,
        )[0]

        # Overwrite transformed velocities with scalar outputs
        # (this is specific to GATr to avoid large jacobians from from log-transforms)
        v_straight[..., self.scalar_dims] = v_s[..., self.scalar_dims]
        return v_straight


class GAPCFM(EventCFMForGA):
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

    def embed_into_ga(self, x, t, ijet):
        # note: ijet is not used
        # (joint training only supported for transformers)

        # scalar embedding
        s = self.t_embedding(t).squeeze()

        # mv embedding
        mv = embed_vector(x.reshape(x.shape[0], -1, 4))
        beam = embed_spurions(
            self.beam_reference,
            self.add_time_reference,
            self.two_beams,
            add_xzplane=False,
            add_yzplane=False,
            device=mv.device,
            dtype=mv.dtype,
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


class GATrCFM(EventCFMForGA):
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
            conceptional keys: coordinates_straight, coordinates_network
        type_token_channels : int
            Number of different particle id's
            Used for one-hot encoding to break permutation symmetry
        process_token_channels : int
            Number of different process id's
            Used for one-hot encoding to break permutation symmetry
        beam_reference : str
            Type of beam reference used to break the Lorentz symmetry
            Options: [None, "xyplane", "spacelike", "lightlike", "timelike"]
            See gatr.interface.spurions.py::embed_spurions for details
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
            This is required whenever coordinates_network != coordinates_straight,
            and the transformation between the two contains e.g. log transforms
        odeint : Dict
            ODE solver settings to be passed to torchdiffeq.odeint
        """
        super().__init__(
            scalar_dims,
            cfm,
            odeint,
        )
        self.net = net
        self.type_token_channels = type_token_channels
        self.process_token_channels = process_token_channels
        self.beam_reference = beam_reference
        self.two_beams = two_beams
        self.add_time_reference = add_time_reference

    def embed_into_ga(self, x, t, ijet):
        # scalar embedding
        type_token = get_type_token(x, self.type_token_channels)
        process_token = get_process_token(x, ijet, self.process_token_channels)
        t_embedding = self.t_embedding(t).expand(x.shape[0], x.shape[1], -1)
        s = torch.cat([type_token, process_token, t_embedding], dim=-1)

        # mv embedding
        mv = embed_vector(x).unsqueeze(-2)
        beam = embed_spurions(
            self.beam_reference,
            self.add_time_reference,
            self.two_beams,
            add_xzplane=False,
            add_yzplane=False,
            device=mv.device,
            dtype=mv.dtype,
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
