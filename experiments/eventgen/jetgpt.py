import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F

import experiments.eventgen.coordinates as c
from experiments.eventgen.coordinates import StandardLogPtPhiEtaLogM2
from experiments.eventgen.wrappers import get_type_token, get_process_token


class CustomMixtureModel:
    # Mixture model with vonMises distribution for angular variables and Normal distribution for others
    # Note that we could also preprocess angles to real space, but then the model would not be periodic
    def __init__(self, n_gauss, channels):
        self.n_gauss = n_gauss
        self.is_angle = torch.tensor(channels) % 4 == 1

    def _extract_logits(self, logits, min_sigmaarg=-10, max_sigmaarg=5):
        logits = logits.reshape(logits.size(0), logits.size(1), self.n_gauss, 3)

        weights = F.softmax(logits[:, :, :, 2], dim=-1)
        mu = logits[:, :, :, 0]

        # avoid inf and 0 (unstable in D.Normal and D.VonMises)
        sigmaarg = torch.clamp(logits[:, :, :, 1], min=min_sigmaarg, max=max_sigmaarg)
        sigma = torch.exp(sigmaarg)
        assert torch.isfinite(sigma).all()

        return mu, sigma, weights

    def build_mm(self, logits):
        mu, sigma, weights = self._extract_logits(logits)
        mix = D.Categorical(weights)
        normal = D.Normal(mu, sigma)
        gmm = D.MixtureSameFamily(mix, normal)
        vonmises = D.VonMises(mu, sigma)
        vmmm = D.MixtureSameFamily(mix, vonmises)
        return gmm, vmmm

    def log_prob(self, x, logits):
        assert x.shape == logits.shape[:-1] and logits.shape[-1] % 3 == 0
        self.is_angle = (
            self.is_angle
            if x.device == self.is_angle.device
            else self.is_angle.to(x.device)
        )
        is_angle = self.is_angle[None, : x.shape[-1]].expand(x.shape)
        gmm, vmmm = self.build_mm(logits)
        log_prob = torch.where(is_angle, vmmm.log_prob(x), gmm.log_prob(x))
        return log_prob

    def sample(self, logits, is_angle):
        gmm, vmmm = self.build_mm(logits)
        mm = vmmm if is_angle else gmm
        x = mm.sample((1,))[:, :, 0].permute(1, 0)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        n_gauss,
        gpt,
    ):
        super().__init__()
        self.mixturemodel = CustomMixtureModel(n_gauss, gpt.channels)
        self.channels = torch.tensor(gpt.channels, dtype=torch.long)

        if gpt.transforms_float64:
            c.DTYPE = torch.float64
        else:
            c.DTYPE = torch.float32

    def init_distribution(self):
        raise NotImplementedError

    def init_coordinates(self):
        raise NotImplementedError

    def get_condition(self, x, idx, ijet):
        raise NotImplementedError

    def _embed_condition(self, x):
        raise NotImplementedError

    def _get_idx(self, shape, device):
        # indices to go back and forth between default (pt, phi, eta, m) ordering
        # and the ordering specified in self.channels
        if self.channels.device != device:
            self.channels = self.channels.to(device)
        channels_mask = self.channels < shape[-1]
        idx = self.channels[channels_mask]
        reverse_idx = torch.argsort(idx)
        return idx, reverse_idx

    def batch_loss(self, x0_fourmomenta, ijet):
        log_prob_gaussian, _ = self._log_prob_gaussian(x0_fourmomenta, ijet)
        loss = -log_prob_gaussian.sum(dim=-1).mean()
        loss_individual = [
            -log_prob_gaussian[:, i::4].sum(dim=-1).mean() for i in range(4)
        ]
        return loss, loss_individual

    def sample(self, ijet, shape, device, dtype, **kwargs):
        shape = (shape[0], shape[1] * shape[2])
        idx, reverse_idx = self._get_idx(shape, device)

        x0_condition = torch.zeros(shape[0], 1, device=device, dtype=dtype)
        for i in range(shape[-1]):
            idx_condition = idx[: i + 1]
            condition = self.get_condition(x0_condition, idx_condition, ijet)
            is_angle = self.channels[i] % 4 == 1
            x0_next = self.mixturemodel.sample(condition[:, [-1], :], is_angle=is_angle)
            x0_condition = torch.cat((x0_condition, x0_next), dim=-1)

        x0_gaussian = x0_condition[:, 1:]
        x0_gaussian = x0_gaussian[:, reverse_idx]
        x0_gaussian = x0_gaussian.reshape(shape[0], shape[1] // 4, 4)
        x0_fourmomenta = self.coordinates_sampling.x_to_fourmomenta(x0_gaussian)
        return x0_fourmomenta

    def _log_prob_gaussian(self, x0_fourmomenta, ijet):
        x0_gaussian = self.coordinates_sampling.fourmomenta_to_x(x0_fourmomenta)
        x0_gaussian = x0_gaussian.reshape(
            x0_gaussian.shape[0], -1
        )  # shape (batchsize, num_components)

        # create idxs from channels
        idx, reverse_idx = self._get_idx(x0_gaussian.shape, x0_gaussian.device)
        x0_gaussian = x0_gaussian[:, idx]

        # create condition and target
        x0_condition = torch.cat(
            (torch.zeros_like(x0_gaussian[:, [0]]), x0_gaussian[..., :-1]), dim=-1
        )
        condition = self.get_condition(x0_condition, idx, ijet)
        log_prob_gaussian = self.mixturemodel.log_prob(x0_gaussian, condition)
        log_prob_gaussian = log_prob_gaussian[:, reverse_idx]
        x0_gaussian = x0_gaussian[:, reverse_idx]
        x0_gaussian = x0_gaussian.reshape(
            x0_gaussian.shape[0], x0_gaussian.shape[1] // 4, 4
        )
        return log_prob_gaussian, x0_gaussian

    def log_prob(self, x0_fourmomenta, ijet):
        log_prob_gaussian, x0_gaussian = self._log_prob_gaussian(x0_fourmomenta, ijet)
        log_prob_gaussian = log_prob_gaussian.sum(dim=-1, keepdims=True)
        logdetjac, _ = self.coordinates_sampling.logdetjac_x_to_fourmomenta(x0_gaussian)
        log_prob_fourmomenta = log_prob_gaussian + logdetjac
        return log_prob_fourmomenta[:, 0]


class EventGPT(GPT):
    # Does not support onshell generation yet
    def __init__(self, n_gauss, gpt):
        super().__init__(n_gauss, gpt)
        self.component_channels = 4

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
        self.units = units
        self.pt_min = pt_min
        self.delta_r_min = delta_r_min
        self.onshell_list = onshell_list
        self.onshell_mass = onshell_mass
        self.base_type = base_type
        self.use_delta_r_min = use_delta_r_min
        self.use_pt_min = use_pt_min

    def init_distribution(self):
        self.distributions = []

    def init_coordinates(self):
        self.coordinates_sampling = StandardLogPtPhiEtaLogM2(
            self.pt_min, self.units, self.onshell_list
        )
        self.coordinates = [self.coordinates_sampling]

    def preprocess(self, fourmomenta):
        fourmomenta = fourmomenta / self.units
        return fourmomenta

    def undo_preprocess(self, fourmomenta):
        fourmomenta = fourmomenta * self.units
        return fourmomenta


class JetGPT(EventGPT):
    def __init__(self, net, n_gauss, gpt, type_token_channels, process_token_channels):
        super().__init__(n_gauss, gpt)
        self.net = net
        self.channel_channels = 4 * type_token_channels
        self.process_token_channels = process_token_channels

    def get_condition(self, x, idx, ijet):
        embedding = self._embed(x, idx, ijet)
        c = self.net(embedding, is_causal=True)
        return c

    def _embed(self, x, idx, ijet):
        channels_embedding = get_type_token(x, self.channel_channels)
        ijet_embedding = get_process_token(x, ijet, self.process_token_channels)
        embedding = torch.cat(
            (x[..., None], channels_embedding, ijet_embedding), dim=-1
        )
        return embedding
