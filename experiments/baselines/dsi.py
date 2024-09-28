import torch
from torch import nn

from experiments.baselines import MLP


def sum_of_numbers_up_to(n):
    return (n * (n + 1)) // 2


class DSI(nn.Module):
    """
    A modification of the MLP to apply a learnable preprocessing on the inputs based on the Deep Sets framework.
    Once the latent space vectors are generated for each particle, a sequence of momentum invariants is concatenated to them and fed to the MLP.
    """

    def __init__(
        self,
        in_shape,
        out_shape,
        num_particles_boson,
        num_particles_glu,
        hidden_channels_prenet,
        hidden_layers_prenet,
        out_dim_prenet_sep,
        hidden_channels_net,
        hidden_layers_net,
    ):
        super().__init__()

        if not hidden_layers_prenet > 0 or not hidden_layers_net > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.num_particles_boson = num_particles_boson
        self.num_particles_glu = num_particles_glu
        self.out_dim_prenet_sep = out_dim_prenet_sep

        self.input_dim_contracted = sum_of_numbers_up_to(
            2 + num_particles_boson + num_particles_glu - 1
        )

        self.prenet_ini = MLP(
            in_shape=4,
            out_shape=out_dim_prenet_sep,  # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
            hidden_channels=hidden_channels_prenet,
            hidden_layers=hidden_layers_prenet,
            dropout_prob=None,
        )

        self.prenet_boson = MLP(
            in_shape=4,
            out_shape=out_dim_prenet_sep,  # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
            hidden_channels=hidden_channels_prenet,
            hidden_layers=hidden_layers_prenet,
            dropout_prob=None,
        )

        self.prenet_jet = MLP(
            in_shape=4,
            out_shape=out_dim_prenet_sep,  # hyperparameter: for instance >num_features,  -> 10, 20, 30, ...? num_features x num_particles
            hidden_channels=hidden_channels_prenet,
            hidden_layers=hidden_layers_prenet,
            dropout_prob=None,
        )

        self.net = MLP(
            in_shape=out_dim_prenet_sep * (2 + num_particles_boson + 1)
            + self.input_dim_contracted,
            out_shape=1,
            hidden_channels=hidden_channels_net,
            hidden_layers=hidden_layers_net,
            dropout_prob=None,
        )

    def forward(self, x):
        n_particles = 2 + self.num_particles_boson + self.num_particles_glu
        particles = x[:, : 4 * n_particles].reshape(-1, n_particles, 4)
        invariants = x[:, 4 * n_particles :]

        deepset_ini = self.prenet_ini(particles[:, :2])
        deepset_ini = deepset_ini.reshape(deepset_ini.shape[0], -1)

        deepset_boson = self.prenet_boson(
            particles[:, 2 : (2 + self.num_particles_boson)]
        )
        deepset_boson = deepset_boson.reshape(deepset_boson.shape[0], -1)

        deepset_jet = self.prenet_jet(
            particles[:, (2 + self.num_particles_boson) :]
        ).sum(dim=-2)

        deepset = torch.cat((deepset_ini, deepset_boson, deepset_jet), 1)

        latent_full = torch.cat((deepset, invariants), 1)

        return self.net(latent_full)
