import torch
from torch import nn
import os
import time
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from hydra.utils import instantiate

from experiments.eventgen.geometry import SimplePossiblyPeriodicGeometry
from experiments.base_plots import plot_loss, plot_metric
from experiments.eventgen.plots import plot_trajectories_over_time, plot_trajectories_2d
from experiments.logger import LOGGER


class MFM(SimplePossiblyPeriodicGeometry):
    def __init__(self, virtual_components, cfm, coordinates, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_components_plot = virtual_components
        self.cfm = cfm
        self.coordinates = coordinates
        self.dnet = None

    def get_loss(self, metrics):
        raise NotImplementedError

    def get_metric(self, y1, y2, x):
        # default: MSE metric
        return super().get_metric(y1, y2, x)

    @torch.enable_grad()
    def get_trajectory(self, x_target, x_base, t):
        # notation: varphi is the trajectory displacement,
        # phi is one of the coordinates
        def get_varphi(t):
            varphi = self.dnet(x_target, x_base, t)
            varphi[..., self.periodic_components] = torch.pi * torch.tanh(
                varphi[..., self.periodic_components]
            )
            return varphi

        t.requires_grad_()
        varphi, dvarphi_dt = torch.autograd.functional.jvp(
            get_varphi,
            t,
            torch.ones_like(t),
            create_graph=True,
            strict=True,
        )

        xt_naive, vt_naive = SimplePossiblyPeriodicGeometry.get_trajectory(
            self, x_target, x_base, t
        )

        xt = xt_naive + t * (1 - t) * varphi
        vt = vt_naive + t * (1 - t) * dvarphi_dt + (1 - 2 * t) * varphi
        xt = self._handle_periodic(xt)
        vt = self._handle_periodic(vt)
        return xt, vt

    def initialize(
        self,
        base,
        target,
        dnet_cfg,
        model_path=None,
        plot_path=None,
        device=None,
        dtype=None,
    ):
        n_features = base.flatten(start_dim=-2).shape[-1]
        self._initialize_model(dnet_cfg, n_features, device, dtype)
        if self.cfm.mfm.warmstart_path is None:
            self._initialize_train(base, target, plot_path, model_path, device, dtype)

        # trajectories plots
        if plot_path is not None:
            if self.cfm.mfm.startup.plot_trajectories:
                os.makedirs(plot_path, exist_ok=True)
                LOGGER.info(f"Starting to create dnet trajectory plots in {plot_path}.")
                filename = os.path.join(plot_path, "dnet_trajectories.pdf")
                with PdfPages(filename) as file:
                    self._plot_trajectories(file, base, target, device, dtype)

    def _initialize_model(self, dnet_cfg, n_features, device, dtype):
        self.dnet = instantiate(
            dnet_cfg,
            n_features=n_features,
            embed_t_dim=self.cfm.embed_t_dim,
            embed_t_scale=self.cfm.embed_t_scale,
        )
        self.dnet = self.dnet.to(device, dtype)
        num_parameters = sum(
            p.numel() for p in self.dnet.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Instantiated dnet {type(self.dnet).__name__} with {num_parameters} learnable parameters."
        )

        # load weights if warmstart is provided
        if self.cfm.mfm.warmstart_path is not None:
            model_path = os.path.join(self.cfm.mfm.warmstart_path, "dnet.pt")
            state_dict = torch.load(model_path, map_location=device)
            self.dnet.load_state_dict(state_dict)
            LOGGER.info(f"Loaded dnet from {self.cfm.mfm.warmstart_path}")

    def _initialize_train(self, base, target, plot_path, model_path, device, dtype):
        LOGGER.info(
            f"Using base/target datasets of shape {tuple(base.shape)}/{tuple(target.shape)}."
        )

        # dataset and loader
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

        # training preparations
        optimizer = torch.optim.Adam(self.dnet.parameters(), lr=self.cfm.mfm.startup.lr)
        metrics = {
            "full": [],
            "full_phi0": [],
            "lr": [],
            "grad_norm": [],
        }
        self._extend_metrics(metrics)
        kwargs = {"optimizer": optimizer, "metrics": metrics}
        loss_min, patience = float("inf"), 0

        # train loop
        t0 = time.time()
        LOGGER.info(
            f"Starting to train dnet for {self.cfm.mfm.startup.iterations} iterations "
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
        LOGGER.info(f"Mean dnet loss: {mean_loss:.2f}")

        # save network
        if model_path is not None:
            os.makedirs(model_path, exist_ok=True)
            model_path = os.path.join(model_path, "dnet.pt")
            torch.save(self.dnet.state_dict(), model_path)
            LOGGER.info(f"Saved dnet to {model_path}")

        # training plots
        if plot_path is not None:
            os.makedirs(plot_path, exist_ok=True)
            LOGGER.info(f"Starting to create dnet training plots in {plot_path}.")
            if self.cfm.mfm.startup.plot_training:
                filename = os.path.join(plot_path, "dnet_training.pdf")
                with PdfPages(filename) as file:
                    self._plot_training(file, metrics)

    def _step(self, x_base, x_target, metrics, optimizer):
        t = torch.rand(x_base.shape[0], 1, 1, device=x_base.device, dtype=x_base.dtype)
        xt, vt = self.get_trajectory(x_target, x_base, t)
        loss, metrics_phi = self._get_loss(xt, vt)
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
        xt_straight, vt_straight = SimplePossiblyPeriodicGeometry.get_trajectory(
            self,
            x_target,
            x_base,
            t,
        )
        loss_phi0, metrics_phi0 = self._get_loss(xt_straight, vt_straight)

        metrics["full"].append(loss.item())
        metrics["full_phi0"].append(loss_phi0.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        metrics["grad_norm"].append(grad_norm)
        for key in metrics_phi.keys():
            metrics[key].append(metrics_phi[key].item())
            metrics[f"{key}_phi0"].append(metrics_phi0[key].item())

        return loss.item()

    def _plot_training(self, file, metrics):
        plot_loss(
            file,
            [metrics["full"], metrics["full_phi0"]],
            metrics["lr"],
            labels=["full loss", r"full loss with $\varphi=0$"],
            logy=False,
        )
        plot_metric(
            file,
            [metrics["grad_norm"]],
            "Gradient norm",
            logy=True,
        )
        metrics_plot, labels = [], []
        for key in metrics.keys():
            if key in ["grad_norm", "lr"]:
                continue
            metrics_plot.append(metrics[key])
            labels.append(key)
        plot_loss(
            file,
            metrics_plot,
            labels=labels,
            logy=False,
        )

    def _create_sample_trajectories(
        self, base, target, device, dtype, nsamples=10, nt=1000
    ):
        t = (
            torch.linspace(0, 1, nt)
            .reshape(-1, 1, 1, 1)
            .repeat(1, nsamples, 1, 1)
            .to(device, dtype)
        )
        x_base = base[None, :nsamples].repeat(nt, 1, 1, 1).to(device, dtype)
        x_target = target[None, :nsamples].repeat(nt, 1, 1, 1).to(device, dtype)
        xt = self.get_trajectory(x_target, x_base, t)[0].detach().cpu()
        xt_straight = (
            SimplePossiblyPeriodicGeometry.get_trajectory(self, x_target, x_base, t)[0]
            .detach()
            .cpu()
        )
        t = t.detach().cpu()
        return xt, xt_straight, t

    def _plot_trajectories_simple(self, file, xt, xt_straight, t, nsamples):
        for i in range(xt.shape[-2]):
            for j in range(xt.shape[-1]):
                plot_trajectories_over_time(
                    file,
                    xt[:, :, i, j].clone(),
                    xt_straight[:, :, i, j].clone(),
                    t[:, :, 0, 0],
                    xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
                    is_phi=j == 1,
                    ylabel=r"$x_{%s}(t)$" % str(4 * i + j),
                )

        xt_fourmomenta = self.coordinates.x_to_fourmomenta(xt)
        xt_straight_fourmomenta = self.coordinates.x_to_fourmomenta(xt_straight)
        for particles in self.virtual_components_plot:
            x_particle = xt_fourmomenta[..., particles, :].sum(dim=-2)
            x_straight_particle = xt_straight_fourmomenta[..., particles, :].sum(dim=-2)
            mass = self._get_mass(x_particle)
            mass_straight = self._get_mass(x_straight_particle)
            plot_trajectories_over_time(
                file,
                mass,
                mass_straight,
                t[:, :, 0, 0],
                xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
                ylabel=r"$m_{\mathrm{{%s}}}(t)$" % particles,
                nmax=nsamples,
            )

        for p1, p2 in [[1, 2], [3, 4], [1, 3], [2, 4]]:
            particles1, particles2 = (
                self.virtual_components_plot[p1],
                self.virtual_components_plot[p2],
            )
            x_particle1 = xt_fourmomenta[..., particles1, :].sum(dim=-2)
            x_particle2 = xt_fourmomenta[..., particles2, :].sum(dim=-2)
            x_straight_particle1 = xt_straight_fourmomenta[..., particles1, :].sum(
                dim=-2
            )
            x_straight_particle2 = xt_straight_fourmomenta[..., particles2, :].sum(
                dim=-2
            )
            mass1 = self._get_mass(x_particle1)
            mass2 = self._get_mass(x_particle2)
            mass1_straight = self._get_mass(x_straight_particle1)
            mass2_straight = self._get_mass(x_straight_particle2)
            plot_trajectories_2d(
                file,
                mass1_straight,
                mass2_straight,
                mass1,
                mass2,
                xlabel=r"$m_{\mathrm{{%s}}}$" % particles1,
                ylabel=r"$m_{\mathrm{{%s}}}$" % particles2,
            )

    def _plot_trajectories(
        self, file, base, target, device, dtype, nsamples=10, nt=1000
    ):
        xt, xt_straight, t = self._create_sample_trajectories(
            base, target, device, dtype, nsamples, nt
        )
        self._plot_trajectories_simple(file, xt, xt_straight, t, nsamples)

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

    def _extend_metrics(self, metrics):
        pass


class MassMFM(MFM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_components_mfm = np.array(self.virtual_components_plot)[
            np.array([1, 2, 3, 4])
        ]

    def _get_loss(self, x, v):
        naive_term = (v**2).sum(dim=[-1, -2]).mean()

        x.requires_grad_()  # required for phi=0 trajectories

        mass_term = []
        x_fourmomenta = self.coordinates.x_to_fourmomenta(x)
        for particles in self.virtual_components_mfm:
            x_particle = x_fourmomenta[..., particles, :].sum(dim=-2)
            mass = self._get_mass(x_particle)[:, None, None]
            grad_outputs = torch.ones_like(mass)
            dmass_dx = torch.autograd.grad(
                mass, x, grad_outputs=grad_outputs, create_graph=True
            )[0]
            mass_term0 = (dmass_dx * v).sum(dim=[-1, -2]) ** 2
            mass_term.append(mass_term0)
        mass = torch.stack(mass_term, dim=-1)
        mass_top = self.cfm.mfm.alpha_top * mass[..., [0, 1]].sum(dim=-1).mean()
        mass_W = self.cfm.mfm.alpha_W * mass[..., [2, 3]].sum(dim=-1).mean()

        loss = naive_term + mass_top + mass_W
        metrics = {"naive": naive_term, "mass_W": mass_W, "mass_top": mass_top}
        return loss, metrics

    def _extend_metrics(self, metrics):
        for key in ["naive", "mass_top", "mass_W"]:
            metrics[key] = []
            metrics[f"{key}_phi0"] = []

    def _plot_training(self, file, metrics):
        super()._plot_training(file, metrics)
        for key in ["naive", "mass_top", "mass_W"]:
            plot_loss(
                file,
                [metrics[key], metrics[f"{key}_phi0"]],
                labels=[key, f"{key} with phi=0"],
                logy=False,
            )

    def _get_distance(self, x1, x2):
        diff = x1 - x2
        diff = self._handle_periodic(diff)
        naive_term = (diff**2).sum(dim=[-1, -2])

        mass_term = []
        x1_fourmomenta = self.coordinates.x_to_fourmomenta(x1)
        x2_fourmomenta = self.coordinates.x_to_fourmomenta(x2)
        for particles in self.virtual_components_mfm:
            x1_particle = x1_fourmomenta[..., particles, :].sum(dim=-2)
            x2_particle = x2_fourmomenta[..., particles, :].sum(dim=-2)
            mass1 = self._get_mass(x1_particle)
            mass2 = self._get_mass(x2_particle)
            mass_term0 = (mass1 - mass2) ** 2
            mass_term.append(mass_term0)
        mass = torch.stack(mass_term, dim=-1)
        mass_top = self.cfm.mfm.alpha_top * mass[..., [0, 1]].sum(dim=-1)
        mass_W = self.cfm.mfm.alpha_W * mass[..., [2, 3]].sum(dim=-1)

        distance = torch.sqrt(naive_term + mass_top + mass_W)
        return distance

    def _plot_trajectories(
        self, file, base, target, device, dtype, nsamples=10, nt=1000
    ):
        xt, xt_straight, t = self._create_sample_trajectories(
            base, target, device, dtype, nsamples, nt
        )

        self._plot_trajectories_simple(file, xt, xt_straight, t, nsamples)
        self._plot_trajectories_distance(
            file, base[:nsamples], target[:nsamples], xt, xt_straight, t, nsamples
        )

    def _plot_trajectories_distance(
        self, file, base, target, xt, xt_straight, t, nsamples
    ):
        xt_base = base.unsqueeze(-4).repeat(xt.shape[-4], 1, 1, 1)
        distance = self._get_distance(xt_base, xt)
        distance_straight = self._get_distance(xt_base, xt_straight)
        distance_max = distance[[0]].clone()
        distance /= distance_max
        distance_straight /= distance_max

        plot_trajectories_over_time(
            file,
            distance,
            distance_straight,
            t[:, :, 0, 0],
            xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
            ylabel=r"rescaled remaining distance to base",
            nmax=nsamples,
        )


class LANDMFM(MFM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = self.cfm.mfm.land.sigma
        self.eps = self.cfm.mfm.land.eps

    def _get_diag_entries(self, x):
        # see (9) in arXiv:2405.14780
        x_emb = x.flatten(start_dim=-2)
        x1, x2 = x_emb.unsqueeze(-2), x_emb.unsqueeze(-3)
        diff = x1 - x2
        diff2 = diff**2
        exponent = -diff2.sum(dim=-1) / (2 * self.sigma**2)
        h = diff2 * torch.exp(exponent.unsqueeze(-1))
        h = h.mean(dim=-3)
        h = h.reshape_as(x)
        return h

    def get_metric(self, y1, y2, x):
        diag_entries = self._get_diag_entries(x)
        diff = y1 - y2
        metric = diff**2 / (diag_entries + self.eps)
        metric = metric.sum(dim=[-1, -2])
        return metric

    def _get_loss(self, x, v):
        diag_entries = self._get_diag_entries(x)
        loss = v**2 / (diag_entries + self.eps)
        loss = loss.sum(dim=[-1, -2]).mean()
        return loss, {}
