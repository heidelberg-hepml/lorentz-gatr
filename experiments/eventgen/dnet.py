import torch
from torch import nn
from experiments.eventgen.cfm import GaussianFourierProjection
from experiments.eventgen.wrappers import get_type_token


class DisplacementNet(nn.Module):
    def __init__(
        self,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__()
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=embed_t_dim,
                scale=embed_t_scale,
            ),
            nn.Linear(embed_t_dim, embed_t_dim),
        )

    def embed_t(self, t):
        return self.t_embedding(t[..., 0])

    def forward(self, x_base, x_target, t):
        raise NotImplementedError


class DisplacementMLP(DisplacementNet):
    def __init__(
        self,
        net,
        n_features,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__(embed_t_dim, embed_t_scale)
        # inputs: x_base (n_features), x_target (n_features), t (embed_t_dim)
        # outputs: phi (n_features)
        self.net = net(
            in_shape=2 * n_features + embed_t_dim,
            out_shape=n_features,
        )

    def embed(self, x_base, x_target, t):
        t_emb = self.embed_t(t)
        x_base_emb = x_base.flatten(start_dim=-2)
        x_target_emb = x_target.flatten(start_dim=-2)
        embedding = torch.cat((x_base_emb, x_target_emb, t_emb), dim=-1)
        return embedding

    def forward(self, x_base, x_target, t):
        embedding = self.embed(x_base, x_target, t)
        phi = self.net(embedding).reshape_as(x_base)
        return phi


class DisplacementTransformer(DisplacementNet):
    def __init__(
        self,
        net,
        n_features,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__(embed_t_dim, embed_t_scale)
        # we use one channel for each particle, instead of one channel for each component
        # inputs per token: x_base (4), x_target (4), t (embed_t_dim), idx (n_particles)
        # outputs per token: phi (4)
        self.n_particles = n_features // 4
        self.net = net(
            in_channels=2 * 4 + embed_t_dim + self.n_particles,
        )

    def embed(self, x_base, x_target, t):
        t_emb = self.embed_t(t)
        t_emb = t_emb.unsqueeze(-2).expand(*x_base.shape[:-1], t_emb.shape[-1])
        type_token = get_type_token(x_base, self.n_particles)
        type_token = type_token.expand(*x_base.shape[:-1], type_token.shape[-1])
        embedding = torch.cat((x_base, x_target, t_emb, type_token), dim=-1)
        return embedding

    def forward(self, x_base, x_target, t):
        embedding = self.embed(x_base, x_target, t)
        phi = self.net(embedding)
        return phi
