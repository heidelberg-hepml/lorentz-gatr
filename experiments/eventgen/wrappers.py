import math
import torch
from torch import nn

from gatr.interface import embed_vector, extract_vector, extract_scalar
from gatr.layers import EquiLinear, GeometricBilinear, ScalarGatedNonlinearity
from experiments.toptagging.dataset import embed_beam_reference
from experiments.eventgen.cfm import EventCFM


def get_type_token(x_ref, type_token_channels):
    type_token_raw = torch.arange(x_ref.shape[1], device=x_ref.device, dtype=torch.long)
    type_token = nn.functional.one_hot(type_token_raw, num_classes=type_token_channels)
    type_token = type_token.unsqueeze(0).expand(
        x_ref.shape[0], x_ref.shape[1], type_token_channels
    )
    return type_token


def get_process_token(x_ref, ijet, process_token_channels):
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


### CFM on 4momenta


class MLPCFM(EventCFM):
    def __init__(
        self,
        net,
        cfm,
        odeint,
    ):
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
    def __init__(
        self,
        net,
        cfm,
        beam_reference,
        two_beams,
        add_time_reference,
        odeint,
    ):
        super().__init__(
            cfm,
            odeint=odeint,
        )
        self.net = net
        self.beam_reference = beam_reference
        self.two_beams = two_beams
        self.add_time_reference = add_time_reference
        assert (
            self.cfm.coordinates_network == "Fourmomenta"
        ), f"GA-networks require coordinates_network=Fourmomenta"

    def get_velocity(self, fourmomenta, t, ijet):
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta

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
            mv = torch.cat([mv, beam], dim=-2)

        return mv, s

    def extract_from_ga(self, mv, s):
        v = extract_vector(mv).squeeze(dim=-2)
        return v


class TransformerCFM(EventCFM):
    def __init__(
        self,
        net,
        cfm,
        type_token_channels,
        process_token_channels,
        odeint,
    ):
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
    Abstract base class for all GATrCFM's
    Add GATr-specific parameters
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
        odeint,
    ):
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
        assert (
            self.cfm.coordinates_network == "Fourmomenta"
        ), f"GA-networks require coordinates_network=Fourmomenta"

    def get_velocity(self, fourmomenta, t, ijet):
        mv, s = self.embed_into_ga(fourmomenta, t, ijet)
        mv_outputs, s_outputs = self.net(mv, s)
        v_fourmomenta = self.extract_from_ga(mv_outputs, s_outputs)
        return v_fourmomenta

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
        return v
