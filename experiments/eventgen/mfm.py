import torch
from torch import nn
import os
import time
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from experiments.eventgen.coordinates import StandardLogPtPhiEtaLogM2
from experiments.eventgen.cfm import GaussianFourierProjection
from experiments.baselines.mlp import MLP
from experiments.base_plots import plot_loss, plot_metric
from experiments.eventgen.plots import plot_trajectories_2d, plot_trajectories_over_time
from experiments.logger import LOGGER


class MassMFM(StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, virtual_components, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfm = cfm
        self.virtual_components = np.array(virtual_components)[
            self.cfm.mfm.virtual_particles
        ]
        self.dnet = None  # displacement net

    def _get_displacement(self, x_base, x_target, t):
        t_emb = self.t_embedding(t[..., 0])
        x_base_emb = x_base.flatten(start_dim=-2)
        x_target_emb = x_target.flatten(start_dim=-2)
        embedding = torch.cat(
            (x_base_emb, x_target_emb, t_emb),
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
            x1_particle = x1_fourmomenta[..., particle, :].sum(dim=-2)
            x2_particle = x2_fourmomenta[..., particle, :].sum(dim=-2)
            m1 = self._get_mass(x1_particle)
            m2 = self._get_mass(x2_particle)
            mass_term0 = 0.5 * ((m1 - m2) ** 2)
            mass_term.append(mass_term0)
        mass_term = torch.stack(mass_term, dim=-1).sum(dim=-1)

        metric = naive_term + self.cfm.mfm.alpha * mass_term
        return metric

    @torch.enable_grad()
    def get_trajectory(self, x_base, x_target, t):
        t.requires_grad_()
        phi = self._get_displacement(x_base, x_target, t)
        xt = x_base + t * (x_target - x_base) + t * (1 - t) * phi

        dphi_dt = []
        for i in range(xt.shape[-2]):
            dphi_dt0 = []
            for j in range(xt.shape[-1]):
                grad_outputs = torch.ones_like(t)
                dphi_dt1 = torch.autograd.grad(
                    phi[..., [i], :][..., [j]],
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

    def initialize(self, base, target, plot_path=None, device=None, dtype=None):
        # initialize dnet
        n_features = base.flatten(start_dim=-2).shape[-1]
        self.t_embedding = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=self.cfm.embed_t_dim,
                scale=self.cfm.embed_t_scale,
            ),
            nn.Linear(self.cfm.embed_t_dim, self.cfm.embed_t_dim),
        ).to(device, dtype)
        self.dnet = MLP(
            in_shape=2 * n_features + self.cfm.embed_t_dim,
            out_shape=n_features,
            hidden_channels=self.cfm.mfm.dnet.hidden_channels,
            hidden_layers=self.cfm.mfm.dnet.hidden_layers,
        ).to(device, dtype)
        num_parameters = sum(
            p.numel() for p in self.dnet.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Instantiated dnet net with {num_parameters} learnable parameters."
        )

        # train dnet
        LOGGER.info(
            f"Using base/target datasets of shape {tuple(base.shape)}/{tuple(target.shape)}."
        )
        dataset = torch.utils.data.TensorDataset
        loader = lambda dataset: torch.utils.data.DataLoader(
            dataset, batch_size=self.cfm.mfm.startup.batchsize, shuffle=True
        )

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        constructor = lambda tensor: iter(cycle(loader(dataset(tensor))))
        iter_base, iter_target = constructor(base), constructor(target)
        optimizer = torch.optim.Adam(self.dnet.parameters(), lr=self.cfm.mfm.startup.lr)
        metrics = {
            "full": [],
            "lr": [],
            "grad_norm": [],
            "naive": [],
            "mass": [],
            "full_phi0": [],
            "naive_phi0": [],
            "mass_phi0": [],
        }
        kwargs = {"optimizer": optimizer, "metrics": metrics}
        loss_min, patience = float("inf"), 0
        t0 = time.time()
        LOGGER.info(
            f"Starting to train dnet net for {self.cfm.mfm.startup.iterations} iterations "
            f"(batchsize={self.cfm.mfm.startup.batchsize}, lr={self.cfm.mfm.startup.lr}, "
            f"patience={self.cfm.mfm.startup.patience})"
        )
        for iteration in range(self.cfm.mfm.startup.iterations):
            x_base = next(iter_base)[0].to(device, dtype)
            x_target = next(iter_target)[0].to(device, dtype)
            loss = self._step(x_base, x_target, **kwargs)

            if loss < loss_min:
                loss_min = loss
                patience = 0
            else:
                patience += 1
                if patience > self.cfm.mfm.startup.patience:
                    break
        dt = time.time() - t0
        LOGGER.info(
            f"Finished training dnet after {iteration} iterations / {dt/60:.2f}min"
        )
        mean_loss = np.mean(metrics["full"][-patience:])
        mean_loss_naive = np.mean(metrics["naive"][-patience:])
        mean_loss_mass = np.mean(metrics["mass"][-patience:])
        LOGGER.info(
            f"Mean dnet loss: {mean_loss:.2f} = {mean_loss_naive:.2f} + {mean_loss_mass:.2f}"
        )

        # create plots
        if plot_path is not None:
            os.makedirs(plot_path, exist_ok=True)
            LOGGER.info(f"Starting to create dnet plots in {plot_path}.")
            if self.cfm.mfm.startup.plot_training:
                filename = os.path.join(plot_path, "dnet_training.pdf")
                with PdfPages(filename) as file:
                    self._training_plots(file, metrics)
            if self.cfm.mfm.startup.plot_trajectories:
                filename = os.path.join(plot_path, "dnet_trajectories.pdf")
                with PdfPages(filename) as file:
                    self._plot_trajectories(file, base, target, device, dtype)

    def _step(self, x_base, x_target, metrics, optimizer):
        t = torch.rand(x_base.shape[0], 1, 1, device=x_base.device, dtype=x_base.dtype)
        xt, vt = self.get_trajectory(x_base, x_target, t)
        loss, loss_naive, loss_mass = self._get_loss(xt, vt)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.dnet.parameters(),
                self.cfm.mfm.startup.clip_grad_norm,
                error_if_nonfinite=True,
            )
            .cpu()
            .item()
        )
        optimizer.step()

        # evaluate loss also for straight trajectories (phi=0)
        xt, vt = super().get_trajectory(x_base, x_target, t)
        loss_phi0, loss_naive_phi0, loss_mass_phi0 = self._get_loss(xt, vt)

        metrics["full"].append(loss.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        metrics["grad_norm"].append(grad_norm)
        metrics["naive"].append(loss_naive.item())
        metrics["mass"].append(loss_mass.item())
        metrics["full_phi0"].append(loss_phi0.item())
        metrics["naive_phi0"].append(loss_naive_phi0.item())
        metrics["mass_phi0"].append(loss_mass_phi0.item())

        return loss.item()

    def _training_plots(self, file, metrics):
        plot_loss(
            file,
            [metrics["full"], metrics["full_phi0"]],
            metrics["lr"],
            labels=["full loss", r"full loss with $\varphi=0$"],
            logy=False,
        )
        plot_loss(
            file,
            [
                metrics["full"],
                metrics["naive"],
                metrics["mass"],
                metrics["full_phi0"],
                metrics["naive_phi0"],
                metrics["mass_phi0"],
            ],
            metrics["lr"],
            labels=[
                "full",
                "naive",
                "mass",
                r"full with $\varphi=0$",
                r"naive with $\varphi=0$",
                r"mass with $\varphi=0$",
            ],
            logy=False,
        )
        plot_metric(
            file,
            [metrics["grad_norm"]],
            "Gradient norm",
            logy=True,
        )

    def _plot_trajectories(
        self, file, base, target, device, dtype, nsamples=10, nt=1000
    ):
        t = (
            torch.linspace(0, 1, nt)
            .reshape(-1, 1, 1, 1)
            .repeat(1, nsamples, 1, 1)
            .to(device, dtype)
        )
        x_base = base[None, :nsamples].repeat(nt, 1, 1, 1).to(device, dtype)
        x_target = target[None, :nsamples].repeat(nt, 1, 1, 1).to(device, dtype)
        xt = self.get_trajectory(x_base, x_target, t)[0].detach().cpu()
        xt_straight = super().get_trajectory(x_base, x_target, t)[0].detach().cpu()
        t = t.detach().cpu()

        for i in range(base.shape[-2]):
            for j in range(base.shape[-1]):
                plot_trajectories_over_time(
                    file,
                    xt[:, :, i, j],
                    xt_straight[:, :, i, j],
                    t[:, :, 0, 0],
                    xlabel=r"$t$",
                    ylabel=r"$x(t)$",
                )

        xt_fourmomenta = self.x_to_fourmomenta(xt)
        xt_straight_fourmomenta = self.x_to_fourmomenta(xt_straight)
        for particles in self.virtual_components:
            x_particle = xt_fourmomenta[..., particles, :].sum(dim=-2)
            x_straight_particle = xt_straight_fourmomenta[..., particles, :].sum(dim=-2)
            mass = self._get_mass(x_particle)
            mass_straight = self._get_mass(x_straight_particle)
            plot_trajectories_over_time(
                file,
                mass,
                mass_straight,
                t[:, :, 0, 0],
                xlabel=r"$t$",
                ylabel=r"$m(t)$",
                nmax=nsamples,
            )

    def _get_mass(self, particle):
        # particle has to be in 'Fourmomenta' format
        unpack = lambda x: [x[..., j] for j in range(4)]
        E, px, py, pz = unpack(particle)
        mass2 = E**2 - px**2 - py**2 - pz**2

        # preprocessing
        prepd = mass2.clamp(min=1e-5) ** 0.5
        if self.cfm.mfm.use_logmass:
            prepd = prepd.log()

        assert torch.isfinite(
            prepd
        ).all(), f"{torch.isnan(prepd).sum()} {torch.isinf(prepd).sum()}"
        return prepd

    def _get_loss(self, x, v):
        naive_term = (v**2).sum(dim=[-1, -2]).mean()

        mass_term = []
        x_fourmomenta = self.x_to_fourmomenta(x)
        for particles in self.virtual_components:
            x_particle = x_fourmomenta[..., particles, :].sum(dim=-2)
            mass = self._get_mass(x_particle)[:, None, None]
            grad_outputs = torch.ones_like(mass)
            dmass_dx = torch.autograd.grad(
                mass, x, grad_outputs=grad_outputs, create_graph=True
            )[0]
            mass_term0 = (dmass_dx * v).sum(dim=[-1, -2]) ** 2
            mass_term.append(mass_term0)
        mass_term = torch.stack(mass_term, dim=-1).sum(dim=-1).mean()
        mass_term *= self.cfm.mfm.alpha

        loss = naive_term + mass_term
        return loss, naive_term, mass_term
