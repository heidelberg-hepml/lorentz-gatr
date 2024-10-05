import torch
from torch import nn
from experiments.baselines.mlp import MLP
from experiments.baselines.transformer import Transformer
from experiments.eventgen.cfm import GaussianFourierProjection


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

    def embed_everything(self, x_base, x_target, t):
        t_emb = self.t_embedding(t[..., 0])
        x_base_emb = x_base.flatten(start_dim=-2)
        x_target_emb = x_target.flatten(start_dim=-2)
        return x_base_emb, x_target_emb, t_emb

    def forward(self, x_base, x_target, t):
        raise NotImplementedError


class DisplacementMLP(DisplacementNet):
    def __init__(
        self,
        hidden_channels,
        hidden_layers,
        n_features,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__(embed_t_dim, embed_t_scale)
        self.net = MLP(
            in_shape=2 * n_features + embed_t_dim,
            out_shape=n_features,
            hidden_channels=hidden_channels,
            hidden_layers=hidden_layers,
        )

    def forward(self, x_base, x_target, t):
        x_base_emb, x_target_emb, t_emb = self.embed_everything(x_base, x_target, t)
        embedding = torch.cat(
            (x_base_emb, x_target_emb, t_emb),
            dim=-1,
        )
        phi = self.net(embedding).reshape_as(x_base)
        return phi


class DisplacementTransformer(DisplacementNet):
    def __init__(
        self,
        hidden_channels,
        num_blocks,
        num_heads,
        n_features,
        embed_t_dim,
        embed_t_scale,
    ):
        super().__init__(embed_t_dim, embed_t_scale)
        self.net = Transformer(
            in_channels=2 + embed_t_dim + n_features,
            out_channels=1,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
        )

    def forward(self, x_base, x_target, t):
        x_base_emb, x_target_emb, t_emb = self.embed_everything(x_base, x_target, t)
        x_base_emb, x_target_emb = x_base_emb.unsqueeze(-1), x_target_emb.unsqueeze(-1)
        t_emb = torch.repeat_interleave(
            t_emb.unsqueeze(-2), x_base_emb.shape[-2], dim=-2
        )
        idx = torch.arange(x_base_emb.shape[-2], device=x_base_emb.device)
        idx_emb = torch.nn.functional.one_hot(idx).repeat(
            [1 for _ in range(len(x_base_emb.shape))]
        )
        idx_emb = idx_emb.repeat(list(x_base_emb.shape[:-2]) + [1, 1])
        embedding = torch.cat((x_base_emb, x_target_emb, t_emb, idx_emb), dim=-1)

        phi = self.net(embedding)
        phi = phi.reshape_as(x_base)
        return phi
