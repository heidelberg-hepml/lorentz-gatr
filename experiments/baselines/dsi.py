import torch
from torch import nn
import numpy as np

from experiments.baselines import MLP


def compute_invariants(particles):
    # compute matrix of all inner products
    inner_product = lambda p1, p2: p1[..., 0] * p2[..., 0] - (
        p1[..., 1:] * p2[..., 1:]
    ).sum(dim=-1)
    invariants_matrix = inner_product(
        particles[..., None, :], particles[..., None, :, :]
    )

    # extract upper triangular part (matrix is symmetric)
    idxs = torch.triu_indices(particles.shape[-2], particles.shape[-2], offset=0)
    invariants = invariants_matrix[..., idxs[0], idxs[1]]
    return invariants


class DSI(nn.Module):
    """
    Deep set + invariant approach
    This network combines implicit bias from permutation invariance (deep sets)
    and Lorentz invariance (Lorentz inner products),
    but in a way that breaks both invariances when combining the two approaches

    There are two types of MLP networks:
    - prenet: MLP (one for each particle type) that processes fourmomenta,
      the results are combined in a deep set to form a permutation-invariant result
      This can be viewed as an optional preprocessing step
    - net: MLP that combines the deep set result with Lorentz invariants
      to extract the final result
    """

    def __init__(
        self,
        type_token_list,
        hidden_channels_prenet,
        hidden_layers_prenet,
        out_dim_prenet_sep,
        hidden_channels_net,
        hidden_layers_net,
        use_deepset=True,
        use_invariants=True,
        dropout_prob=None,
    ):
        """
        Parameters
        ----------
        type_token_list : List[int]
            List of particles in the process, with an integer representing the particle type
            Example: [0,0,1,2,2] for q q > Z g g
        hidden_channels_prenet : int
        hidden_layers_prenet: int
        out_dim_prenet_sep : int
            Size of the latent space extract from the deep set
        hidden_channels_net : int
        hidden_layers_net : int
        use_deepset : bool
            whether to use the deep set part (affects prenet)
        use_invariants : bool
            whether to use the invariants part (affects net)
        dropout_prob : float
        """
        super().__init__()
        self.use_deepset = use_deepset
        self.use_invariants = use_invariants
        n = len(type_token_list)
        if self.use_deepset:
            assert len(np.unique(type_token_list)) == max(type_token_list) + 1

            self.prenets = nn.ModuleList(
                [
                    MLP(
                        in_shape=4,
                        out_shape=out_dim_prenet_sep,
                        hidden_channels=hidden_channels_prenet,
                        hidden_layers=hidden_layers_prenet,
                        dropout_prob=dropout_prob,
                    )
                    for _ in range(max(type_token_list) + 1)
                ]
            )
            mlp_inputs = out_dim_prenet_sep * n
        else:
            mlp_inputs = 0

        if self.use_invariants:
            mlp_inputs += n * (n + 1) // 2

        self.net = MLP(
            in_shape=mlp_inputs,
            out_shape=1,
            hidden_channels=hidden_channels_net,
            hidden_layers=hidden_layers_net,
            dropout_prob=dropout_prob,
        )

    def forward(self, particles, type_token):
        # deep set preprocessing
        if self.use_deepset:
            deep_set = []
            for i, type_token_i in enumerate(type_token[0]):
                element = self.prenets[type_token_i](particles[:, i])
                deep_set.append(element)
            deep_set = torch.cat(deep_set, dim=-1)
        else:
            deep_set = torch.empty(
                particles.shape[0], 0, device=particles.device, dtype=particles.dtype
            )

        # invariants
        if self.use_invariants:
            invariants = compute_invariants(particles)
        else:
            invariants = torch.empty(
                particles.shape[0],
                0,
                device=particles.device,
                dtype=particles.dtype,
            )

        # combine everything
        latent_full = torch.cat((deep_set, invariants), dim=-1)
        result = self.net(latent_full)
        return result
