import torch
from torch import nn

from experiments.eventgen.coordinates import StandardLogPtPhiEtaLogM2
from experiments.eventgen.cfm import GaussianFourierProjection
from experiments.baselines.mlp import MLP


class MassMFM(StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, virtual_components, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfm = cfm
        self.virtual_components = virtual_components
        self.dnet = None  # displacement net

    def _get_displacement(self, x_base, x_target, t):
        t_embedding = self.t_embedding(t[..., 0])
        embedding = torch.cat(
            (x_base.flatten(start_dim=-2), x_target.flatten(start_dim=-2), t_embedding),
            dim=-1,
        )
        phi = self.dnet(embedding).reshape_as(x_base)
        return phi

    def get_metric(self, x1, x2):
        naive_term = 0.5 * ((x1 - x2) ** 2).sum(dim=[-1, -2])

        x1_fourmomenta = self.x_to_fourmomenta(x1)
        x2_fourmomenta = self.x_to_fourmomenta(x2)
        mass_term = []
        for particle in self.virtual_components:
            x1_particle = x1_fourmomenta[:, particle, :].sum(dim=-2)
            x2_particle = x2_fourmomenta[:, particle, :].sum(dim=-2)
            m1 = self._get_mass(x1_particle)
            m2 = self._get_mass(x2_particle)
            mass_term0 = 0.5 * ((m1 - m2) ** 2)
            mass_term.append(mass_term0)
        mass_term = torch.stack(mass_term, dim=-1).sum(dim=-1)

        metric = naive_term + self.cfm.mfm.alpha * mass_term
        return metric

    def get_trajectory(self, x_base, x_target, t):
        t.requires_grad_()
        phi = self._get_displacement(x_base, x_target, t)
        xt = x_base + t * (x_target - x_base) + t * (1 - t) * phi

        dphi_dt = []
        for i in range(xt.shape[1]):
            dphi_dt0 = []
            for j in range(xt.shape[2]):
                grad_outputs = torch.ones_like(t)
                dphi_dt1 = torch.autograd.grad(
                    phi[:, [i]][..., [j]],
                    t,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                )[0]
                dphi_dt0.append(dphi_dt1)
            dphi_dt0 = torch.cat(dphi_dt0, dim=-1)
            dphi_dt.append(dphi_dt0)
        dphi_dt = torch.cat(dphi_dt, dim=-2)
        vt = x_target - x_base + t * (1 - t) * dphi_dt + (1 - 2 * t) * phi
        return xt, vt

    def initialize(self, base, target):
        # initialize dnet
        n_features = base.flatten(start_dim=-2).shape[-1]
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=self.cfm.embed_t_dim,
                scale=self.cfm.embed_t_scale,
            ),
            nn.Linear(self.cfm.embed_t_dim, self.cfm.embed_t_dim),
        )
        self.dnet = MLP(
            in_shape=2 * n_features + self.cfm.embed_t_dim,
            out_shape=n_features,
            hidden_channels=self.cfm.mfm.dnet.hidden_channels,
            hidden_layers=self.cfm.mfm.dnet.hidden_layers,
        )

        # train dnet
        dataset = torch.utils.data.TensorDataset(base, target)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfm.mfm.startup.batchsize, shuffle=True
        )

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        iterator = iter(cycle(loader))
        optimizer = torch.optim.Adam(self.dnet.parameters(), lr=self.cfm.mfm.startup.lr)
        for iteration in range(self.cfm.mfm.startup.iterations):
            x_base, x_target = next(iterator)
            t = torch.rand(
                x_base.shape[0], 1, 1, device=x_base.device, dtype=x_base.dtype
            )
            xt, vt = self.get_trajectory(x_base, x_target, t)
            loss, loss_naive, loss_mass = self._get_loss(xt, vt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

    def _get_mass(self, particle):
        # particle has to be in 'Fourmomenta' format
        unpack = lambda x: [x[..., j] for j in range(4)]
        E, px, py, pz = unpack(particle)
        mass2 = E**2 - px**2 - py**2 - pz**2

        # preprocessing
        prepd = mass2**0.5
        if self.cfm.mfm.use_logmass:
            prepd = prepd.log()

        assert torch.isfinite(prepd).all()
        return prepd

    def _get_loss(self, x, v):
        naive_term = (x**2).sum(dim=[-1, -2]).mean()

        mass_term = []
        x_fourmomenta = self.x_to_fourmomenta(x)
        for particles in self.virtual_components:
            x_particle = x_fourmomenta[:, particles, :].sum(dim=-2)
            m = self._get_mass(x_particle)[:, None, None]
            grad_outputs = torch.ones_like(m)
            dm_dx = torch.autograd.grad(
                m, x, grad_outputs=grad_outputs, create_graph=True
            )[0]
            mass_term0 = (dm_dx * x).sum(dim=[-1, -2])
            mass_term.append(mass_term0)
        mass_term = torch.stack(mass_term, dim=-1).sum(dim=-1).mean()

        loss = naive_term + self.cfm.mfm.alpha * mass_term
        return loss, naive_term, mass_term
