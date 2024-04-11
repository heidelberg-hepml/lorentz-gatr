import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict
from tqdm import trange, tqdm

from experiments.base_experiment import BaseExperiment
from experiments.eventgen.dataset import EventDataset, EventDataLoader
from experiments.eventgen.helpers import (
    ensure_onshell,
)
import experiments.eventgen.plotter as plotter
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow


class EventGenerationExperiment(BaseExperiment):
    def init_physics(self):
        self.define_process_specifics()

        # dynamically set wrapper properties
        self.modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        n_particles = self.n_hard_particles + max(self.cfg.data.n_jets)
        n_datasets = len(self.cfg.data.n_jets)
        with open_dict(self.cfg):
            # preparation for joint training
            if self.modelname in ["GATr", "Transformer"]:
                self.cfg.model.process_token_channels = n_datasets
                self.cfg.model.type_token_channels = n_particles
            else:
                # no joint training possible
                assert len(self.cfg.data.n_jets) == 1

            # dynamically set channel dimensions
            if self.modelname == "GATr":
                self.cfg.model.net.in_s_channels = (
                    n_particles + n_datasets + self.cfg.model.embed_t_dim
                )
            elif self.modelname == "Transformer":
                self.cfg.model.net.in_channels = (
                    4 + n_datasets + n_particles + self.cfg.model.embed_t_dim
                )
            elif self.modelname == "GAP":
                self.cfg.model.net.in_mv_channels = n_particles
                self.cfg.model.net.out_mv_channels = n_particles
                self.cfg.model.net.in_s_channels = self.cfg.model.embed_t_dim
            elif self.modelname == "MLP":
                self.cfg.model.net.in_shape = (
                    4 * n_particles + self.cfg.model.embed_t_dim
                )
                self.cfg.model.net.out_shape = 4 * n_particles

            # extra treatment for lorentz-symmetry breaking inputs in equivariant models
            if self.modelname in ["GATr", "GAP"]:
                if self.cfg.model.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += (
                        2 if self.cfg.model.two_beams else 1
                    )
                if self.cfg.model.add_time_reference:
                    self.cfg.model.net.in_mv_channels += 1

            self.cfg.model.odeint_kwargs = self.cfg.odeint_kwargs

    def init_data(self):
        LOGGER.info(f"Working with {self.cfg.data.n_jets} extra jets")

        # load all datasets and organize them in lists
        self.events_raw = []
        for n_jets in self.cfg.data.n_jets:
            # load data
            data_path = eval(f"self.cfg.data.data_path_{n_jets}j")
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            data_raw = np.load(data_path)
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

            # bring data into correct shape
            if self.cfg.data.subsample is not None:
                assert self.cfg.data.subsample < data_raw.shape[0]
                LOGGER.info(
                    f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}"
                )
                data_raw = data_raw[: self.cfg.data.subsample, :]
            data_raw = data_raw.reshape(data_raw.shape[0], data_raw.shape[1] // 4, 4)
            data_raw = torch.tensor(data_raw, dtype=self.dtype)

            # collect everything
            self.events_raw.append(data_raw)

        # change global units
        self.model.init_physics(
            self.units,
            self.pt_min,
            self.delta_r_min,
            self.onshell_list,
            self.onshell_mass,
            self.base_kwargs,
            self.cfg.data.base_type,
            self.cfg.data.use_pt_min,
            self.cfg.data.use_delta_r_min,
            self.cfg.data.mass_scale,
        )
        self.model.init_distribution()
        self.model.init_coordinates()

        # preprocessing
        self.events_prepd = []
        for ijet in range(len(self.cfg.data.n_jets)):
            # preprocess data
            self.events_raw[ijet] = ensure_onshell(
                self.events_raw[ijet], self.onshell_list, self.onshell_mass
            )
            data_prepd = self.model.preprocess(self.events_raw[ijet])
            self.events_prepd.append(data_prepd)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets for each dataset
        train_sets, test_sets, val_sets = [], [], []
        self.data_raw, self.data_prepd = [], []
        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            n_data = self.events_raw[ijet].shape[0]
            split_train = int(n_data * self.cfg.data.train_test_val[0])
            split_test = int(n_data * sum(self.cfg.data.train_test_val[:2]))
            split_val = int(n_data * sum(self.cfg.data.train_test_val))

            data_raw = {
                "trn": self.events_raw[ijet][0:split_train],
                "tst": self.events_raw[ijet][split_train:split_test],
                "val": self.events_raw[ijet][split_test:split_val],
            }
            data_prepd = {
                "trn": self.events_prepd[ijet][0:split_train],
                "tst": self.events_prepd[ijet][split_train:split_test],
                "val": self.events_prepd[ijet][split_test:split_val],
            }
            self.data_raw.append(data_raw)
            self.data_prepd.append(data_prepd)

        # create dataloaders
        self.train_loader = EventDataLoader(
            dataset=EventDataset([x["trn"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = EventDataLoader(
            dataset=EventDataset([x["tst"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = EventDataLoader(
            dataset=EventDataset([x["val"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    @torch.no_grad()
    def evaluate(self):
        self._sample_events()

        # EMA-evaluation not implemented
        loaders = {
            "train": self.train_loader,
            "test": self.test_loader,
            "val": self.val_loader,
            "gen": self.sample_loader,
        }
        for key in self.cfg.evaluation.eval_loss:
            self._evaluate_loss_single(loaders[key], key)
        for key in self.cfg.evaluation.eval_log_prob:
            if key == "gen":
                # log_probs of generated events are not interesting
                # + they are not well-defined, because generated events might be in regions
                # that are not included in the base distribution (because of pt_min, delta_r_min)
                continue
            self._evaluate_log_prob_single(loaders[key], key)

    def _evaluate_loss_single(self, loader, title):
        # use the same random numbers for all datasets to get comparable results
        gen = torch.Generator().manual_seed(42)

        self.model.eval()
        losses = []
        mses = {f"{n_jets}j": [] for n_jets in self.cfg.data.n_jets}
        LOGGER.info(f"Starting to evaluate loss for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(loader):
            loss = 0.0
            for ijet, data_single in enumerate(data):
                x0 = data_single.to(self.device)
                loss_single = self.model.batch_loss(x0, ijet)[0]
                loss += loss_single / len(self.cfg.data.n_jets)
                mses[f"{self.cfg.data.n_jets[ijet]}j"].append(loss_single.cpu().item())
            losses.append(loss.cpu().item())
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating loss for {title} dataset after {dt/60:.2f}min"
        )

        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.loss", np.mean(losses))
            for key, values in mses.items():
                log_mlflow(f"eval.{title}.{key}.mse", np.mean(values))

    def _evaluate_log_prob_single(self, loader, title):
        self.model.eval()
        log_probs = {f"{n_jets}j": [] for n_jets in self.cfg.data.n_jets}
        LOGGER.info(f"Starting to evaluate log_prob for model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(tqdm(loader)):
            for ijet, data_single in enumerate(data):
                x0 = data_single.to(self.device)
                log_prob = self.model.log_prob(x0, ijet).squeeze().cpu()
                log_probs[f"{self.cfg.data.n_jets[ijet]}j"].extend(
                    log_prob.numpy().tolist()
                )
        dt = time.time() - t0
        LOGGER.info(
            f"Finished evaluating log_prob for {title} dataset after {dt/60:.2f}min"
        )
        for key, values in log_probs.items():
            LOGGER.info(f"log_prob_{key} = {np.mean(values)} +- {np.std(values)}")
            if self.cfg.use_mlflow:
                log_mlflow(f"eval.{title}.{key}.log_prob", np.mean(values))
                log_mlflow(f"eval.{title}.{key}.log_prob_std", np.std(values))

    def _sample_events(self):
        self.model.eval()

        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            if self.cfg.save and self.cfg.plotting.save_trajectories:
                os.makedirs(
                    os.path.join(self.cfg.run_dir, "trajectories"), exist_ok=True
                )
                trajectory_path = os.path.join(
                    self.cfg.run_dir,
                    "trajectories",
                    f"run{self.cfg.run_idx}_{n_jets}j.npz",
                )
                LOGGER.info(f"Will save {n_jets}j trajectories to {trajectory_path}")
            else:
                trajectory_path = None

            sample = []
            shape = (self.cfg.evaluation.batchsize, self.n_hard_particles + n_jets, 4)
            n_batches = (
                1 + (self.cfg.evaluation.nsamples - 1) // self.cfg.evaluation.batchsize
            )
            LOGGER.info(
                f"Starting to generate {self.cfg.evaluation.nsamples} {n_jets}j events"
            )
            t0 = time.time()
            for i in trange(n_batches, desc="Sampled batches"):
                x_t = self.model.sample(
                    ijet,
                    shape,
                    self.device,
                    self.dtype,
                    trajectory_path=trajectory_path if i == 0 else None,
                )
                sample.append(x_t)
            t1 = time.time()
            LOGGER.info(
                f"Finished generating {n_jets}j events after {t1-t0:.2f}s = {(t1-t0)/60:.2f}min"
            )

            samples = torch.cat(sample, dim=0)[
                : self.cfg.evaluation.nsamples, ...
            ].cpu()
            self.data_prepd[ijet]["gen"] = samples

            samples_raw = self.model.undo_preprocess(samples)
            self.data_raw[ijet]["gen"] = samples_raw

            m2 = samples_raw[..., 0] ** 2 - (samples_raw[..., 1:] ** 2).sum(dim=-1)
            LOGGER.info(f"Fraction of events with m2<0: {(m2<0).float().mean():.4f}")

        self.sample_loader = torch.utils.data.DataLoader(
            dataset=EventDataset([x["gen"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

    def plot(self):
        path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(os.path.join(path), exist_ok=True)
        LOGGER.info(f"Creating plots in {path}")

        self.plot_titles = [
            r"${%s}+{%s}j$" % (self.plot_title, n_jets)
            for n_jets in self.cfg.data.n_jets
        ]
        kwargs = {
            "exp": self,
            "model_label": self.modelname,
        }

        if self.cfg.plotting.loss and self.cfg.train:
            filename = os.path.join(path, "loss.pdf")
            plotter.plot_losses(filename=filename, **kwargs)

        if self.cfg.plotting.fourmomenta:
            filename = os.path.join(path, "fourmomenta.pdf")
            plotter.plot_fourmomenta(filename=filename, **kwargs)

        if self.cfg.plotting.jetmomenta:
            filename = os.path.join(path, "jetmomenta.pdf")
            plotter.plot_jetmomenta(filename=filename, **kwargs)

        if self.cfg.plotting.preprocessed:
            filename = os.path.join(path, "preprocessed.pdf")
            plotter.plot_preprocessed(filename=filename, **kwargs)

        if self.cfg.plotting.virtual:
            filename = os.path.join(path, "virtual.pdf")
            plotter.plot_virtual(filename=filename, **kwargs)

        if self.cfg.plotting.delta:
            filename = os.path.join(path, "delta.pdf")
            plotter.plot_delta(filename=filename, **kwargs)

        if self.cfg.plotting.deta_dphi:
            filename = os.path.join(path, "deta_dphi.pdf")
            plotter.plot_deta_dphi(filename=filename, **kwargs)

    def _init_loss(self):
        # loss defined manually within the model
        pass

    def _batch_loss(self, data):

        # average over contributions from different datasets
        loss = 0.0
        mse = []
        component_mse = []
        for ijet, x0 in enumerate(data):
            x0 = x0.to(self.device)
            mse_single, component_mse_single = self.model.batch_loss(x0, ijet)
            loss += mse_single / len(self.cfg.data.n_jets)
            mse.append(mse_single.cpu().item())
            component_mse.append([x.cpu().item() for x in component_mse_single])
        assert torch.isfinite(loss).all()

        metrics = {
            f"{n_jets}j.mse": mse[ijet]
            for (ijet, n_jets) in enumerate(self.cfg.data.n_jets)
        }
        for k in range(4):
            for ijet, n_jets in enumerate(self.cfg.data.n_jets):
                metrics[f"{n_jets}j.mse_{k}"] = component_mse[ijet][k]
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{n_jets}j.mse": [] for n_jets in self.cfg.data.n_jets}
        for k in range(4):
            for n_jets in self.cfg.data.n_jets:
                metrics[f"{n_jets}j.mse_{k}"] = []
        return metrics

    def define_process_specifics(self):
        self.plot_title = None
        self.n_hard_particles = None
        self.n_jets_max = None
        self.onshell_list = None
        self.onshell_mass = None
        self.units = None
        self.base_kwargs = None
        self.pt_min = None
        self.delta_r_min = None
        self.obs_names_index = None
        self.fourmomentum_ranges = None
        self.jetmomentum_ranges = None
        self.virtual_components = None
        self.virtual_names = None
        self.virtual_ranges = None
        raise NotImplementedError
