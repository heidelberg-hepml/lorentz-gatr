import torch
from torch import nn
import os
import time
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from hydra.utils import instantiate

from experiments.eventgen.geometry import SimplePossiblyPeriodicGeometry
from experiments.base_plots import plot_loss, plot_metric
from experiments.eventgen.plots import (
    plot_trajectories_over_time,
    plot_trajectories_2d,
    plot_trajectories_straightness,
)
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
        basesampler,
        target,
        dnet_cfg,
        model_path=None,
        plot_path=None,
        device=None,
        dtype=None,
    ):
        n_features = target.flatten(start_dim=-2).shape[-1]
        self._initialize_model(dnet_cfg, n_features, device, dtype)
        train_iter, val_loader, test_target = self._initialize_data(target)
        if self.cfm.mfm.warmstart_path is None:
            self._initialize_train(
                basesampler,
                train_iter,
                val_loader,
                plot_path,
                model_path,
                device,
                dtype,
            )

        # trajectories plots
        if plot_path is not None:
            if self.cfm.mfm.plot_trajectories:
                os.makedirs(plot_path, exist_ok=True)
                LOGGER.info(f"Starting to create dnet trajectory plots in {plot_path}.")
                filename = os.path.join(plot_path, "dnet_trajectories.pdf")
                with PdfPages(filename) as file:
                    self._plot_trajectories(
                        file, basesampler, test_target, device, dtype
                    )

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

    def _initialize_data(
        self,
        target,
    ):
        LOGGER.info(f"Using target dataset of shape {tuple(target.shape)}.")

        splits = [
            int(x * target.shape[0]) for x in self.cfm.mfm.training.train_test_val
        ]
        splits[-1] = target.shape[0] - sum(splits[:-1])
        train, test, val = torch.split(target, splits)

        # dataset and loader
        dataset = torch.utils.data.TensorDataset
        loader = lambda dataset, shuffle: torch.utils.data.DataLoader(
            dataset, batch_size=self.cfm.mfm.training.batchsize, shuffle=shuffle
        )

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        constructor = lambda tensor: iter(cycle(loader(dataset(tensor), shuffle=True)))
        train_iter = constructor(train)
        val_loader = loader(dataset(val), shuffle=False)

        # return objects for further use
        # train_iter: infinite iterator over training data
        # val_loader: validation data loader
        # test: raw test data
        return train_iter, val_loader, test

    def _initialize_train(
        self, basesampler, train_iter, val_loader, plot_path, model_path, device, dtype
    ):
        # training preparations
        optimizer = torch.optim.Adam(
            self.dnet.parameters(), lr=self.cfm.mfm.training.lr
        )
        if self.cfm.mfm.training.scheduler is None:
            scheduler = None
        elif self.cfm.mfm.training.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.cfm.mfm.training.reduceplateau_factor,
                patience=self.cfm.mfm.training.reduceplateau_patience,
            )
        elif self.cfm.mfm.training.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cfm.mfm.training.iterations,
                eta_min=self.cfm.mfm.training.cosanneal_eta_min,
            )
        else:
            raise ValueError(
                f"Scheduler {self.cfm.mfm.training.scheduler} not implemented"
            )
        metrics = {
            "full": [],
            "full_phi0": [],
            "lr": [],
            "grad_norm": [],
            "val_loss": [],
        }
        self._extend_metrics(metrics)
        kwargs = {"optimizer": optimizer, "scheduler": scheduler, "metrics": metrics}
        smallest_val_loss, patience = float("inf"), 0

        # train loop
        t0 = time.time()
        val_time = 0.0
        LOGGER.info(
            f"Starting to train dnet for {self.cfm.mfm.training.iterations} iterations "
            f"(batchsize={self.cfm.mfm.training.batchsize}, lr={self.cfm.mfm.training.lr}, "
            f"patience={self.cfm.mfm.training.es_patience})"
        )
        for step in range(self.cfm.mfm.training.iterations):
            x_target = next(train_iter)[0].to(device, dtype)
            x_base = basesampler(x_target.shape, device, dtype)
            self._step(x_base, x_target, **kwargs)

            if (step + 1) % self.cfm.mfm.training.validate_every_n_steps == 0:
                ta = time.time()
                val_loss = self._validate(val_loader, basesampler, device, dtype)
                metrics["val_loss"].append(val_loss)
                if val_loss < smallest_val_loss:
                    smallest_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1
                    if patience > self.cfm.mfm.training.es_patience:
                        break
                if self.cfm.mfm.training.scheduler in ["ReduceLROnPlateau"]:
                    scheduler.step(val_loss)
                val_time += time.time() - ta

            if step == 1000:
                dt = time.time() - t0
                dtEst = dt * self.cfm.mfm.training.iterations / 1000
                LOGGER.info(
                    f"Finished iteration 1000 after {dt/60:.2f}min, training time estimate: {dtEst:.2f}h"
                )
        dt = time.time() - t0
        LOGGER.info(
            f"Finished training dnet after {step} iterations / {dt/60**2:.2f}h (spent fraction {val_time/dt:.2f} validating)"
        )
        mean_loss = np.mean(metrics["full"])
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
            if self.cfm.mfm.plot_training:
                filename = os.path.join(plot_path, "dnet_training.pdf")
                with PdfPages(filename) as file:
                    self._plot_training(file, metrics)

    def _validate(self, val_loader, basesampler, device, dtype):
        self.dnet.eval()
        losses = []
        with torch.no_grad():
            for (x_target,) in val_loader:
                x_target = x_target.to(device, dtype)
                x_base = basesampler(x_target.shape, device, dtype)
                t = torch.rand(x_base.shape[0], 1, 1, device=device, dtype=dtype)
                xt, vt = self.get_trajectory(x_target, x_base, t)
                loss = self._get_loss(xt, vt)[0]
                losses.append(loss.detach().cpu().item())
        return np.mean(losses)

    def _step(self, x_base, x_target, metrics, optimizer, scheduler):
        t = torch.rand(x_base.shape[0], 1, 1, device=x_base.device, dtype=x_base.dtype)
        xt, vt = self.get_trajectory(x_target, x_base, t)
        loss, metrics_phi = self._get_loss(xt, vt)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                self.dnet.parameters(),
                self.cfm.mfm.training.clip_grad_norm,
                error_if_nonfinite=True,
            )
            .cpu()
            .item()
        )
        optimizer.step()
        if self.cfm.mfm.training.scheduler in ["CosineAnnealingLR"]:
            scheduler.step()

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
            [metrics["full"], metrics["val_loss"]],
            metrics["lr"],
            labels=["train", r"validation"],
            logy=False,
        )
        plot_loss(
            file,
            [metrics["full"], metrics["full_phi0"]],
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
        return base.cpu(), xt, xt_straight, t

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
        self, file, basesampler, target, device, dtype, nsamples=10, nt=1000
    ):
        base = basesampler(target.shape, device, dtype, use_seed=True)
        base, xt, xt_straight, t = self._create_sample_trajectories(
            base, target, device, dtype, nsamples, nt
        )
        self._plot_trajectories_simple(file, xt, xt_straight, t, nsamples)
        return base, xt, xt_straight, t

    def _get_mass(self, particle):
        # particle has to be in 'Fourmomenta' format
        unpack = lambda x: [x[..., j] for j in range(4)]
        E, px, py, pz = unpack(particle)
        mass2 = E**2 - px**2 - py**2 - pz**2

        # preprocessing
        prepd = mass2.clamp(min=1e-5) ** 0.5
        if self.cfm.mfm.use_logmass:
            prepd = prepd.log()

        assert torch.isfinite(prepd).all()
        return prepd

    def _extend_metrics(self, metrics):
        pass


class MassMFM(MFM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_components_mfm = np.array(self.virtual_components_plot)[
            np.array([1, 2, 3, 4])
        ]
        self.alpha_top = self.cfm.mfm.mass.alpha_top
        self.alpha_W = self.cfm.mfm.mass.alpha_W

    @torch.enable_grad()
    def _get_loss(self, x, v):
        scale = x.shape[-2] * x.shape[-1]
        naive_term = (v**2).sum(dim=[-1, -2])

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
        mass_top = self.alpha_top * mass[..., [0, 1]].sum(dim=-1)
        mass_W = self.alpha_W * mass[..., [2, 3]].sum(dim=-1)

        naive_term, mass_top, mass_W = (
            naive_term.mean() / scale,
            mass_top.mean() / scale,
            mass_W.mean() / scale,
        )
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
        mass_top = self.alpha_top * mass[..., [0, 1]].sum(dim=-1)
        mass_W = self.alpha_W * mass[..., [2, 3]].sum(dim=-1)

        distance = torch.sqrt(naive_term + mass_top + mass_W)
        return distance

    def _plot_trajectories(
        self, file, base, target, device, dtype, nsamples=100, nt=1000
    ):
        base, xt, xt_straight, t = super()._plot_trajectories(
            file, base, target, device, dtype, nsamples=100, nt=1000
        )
        self._plot_trajectories_distance(file, base[:nsamples], xt, xt_straight, t)

    def _plot_trajectories_distance(self, file, base, xt, xt_straight, t):
        xt_base = base.unsqueeze(-4).repeat(xt.shape[-4], 1, 1, 1)
        distance = self._get_distance(xt_base, xt)
        distance_straight = self._get_distance(xt_base, xt_straight)
        distance_max = distance[[0]].clone()
        distance /= distance_max
        distance_straight /= distance_max

        t = t[:, :, 0, 0]
        plot_trajectories_over_time(
            file,
            distance,
            distance_straight,
            t,
            xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
            ylabel=r"rescaled remaining distance to base",
        )
        plot_trajectories_straightness(
            file,
            distance,
            t,
            1 - t,
            xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
            ylabel=r"$g$",
        )
        plot_trajectories_straightness(
            file,
            distance / (1 - t),
            t,
            1 + 0.0 * t,
            xlabel=r"$t$ ($t=0$: target, $t=1$: base)",
            ylabel=r"$g/(1-t)$",
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
        exponent = -diff2.mean(dim=-1) / (2 * self.sigma**2)
        h = diff2 * torch.exp(exponent.unsqueeze(-1))
        h = h.mean(dim=-3)
        h = h.reshape_as(x)
        return h

    def get_metric(self, y1, y2, x):
        diag_entries = self._get_diag_entries(x)
        diff = y1 - y2
        metric = diff**2 / (diag_entries + self.eps)
        metric = metric.mean(dim=[-1, -2])
        return metric

    def _get_loss(self, x, v):
        diag_entries = self._get_diag_entries(x)
        loss = v**2 / (diag_entries + self.eps)
        loss = loss.mean(dim=[-1, -2]).mean()
        return loss, {}
