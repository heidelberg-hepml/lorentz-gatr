import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict

from experiments.base_experiment import BaseExperiment
from experiments.eventgen.dataset import EventDataset
from experiments.eventgen.preprocessing import (
    preprocess_gatr,
    undo_preprocess_gatr,
    preprocess_tr,
    undo_preprocess_tr,
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
        with open_dict(self.cfg):
            self.cfg.model.type_token_channels = self.n_hard_particles + max(
                self.cfg.data.n_jets
            )
            self.cfg.model.process_token_channels = len(self.cfg.data.n_jets)
            if self.modelname == "Transformer":
                self.cfg.model.net.in_channels = (
                    4
                    + self.cfg.model.type_token_channels
                    + self.cfg.model.process_token_channels
                    + self.cfg.model.embed_t_dim
                )
            elif self.modelname == "GATr":
                self.cfg.model.net.in_s_channels = (
                    self.cfg.model.type_token_channels
                    + self.cfg.model.process_token_channels
                    + self.cfg.model.embed_t_dim
                )

            if self.modelname == "GATr" and self.cfg.model.beam_reference is not None:
                self.cfg.model.net.in_mv_channels += (
                    2 if self.cfg.model.beam_reference == "cgenn" else 1
                )

            self.cfg.model.onshell_list = self.onshell_list
            self.cfg.model.onshell_mass = self.onshell_mass
            if self.modelname == "GATr":
                self.cfg.model.pt_min = self.pt_min
            self.pt_min = torch.tensor(self.pt_min).unsqueeze(0)

    def init_data(self):
        LOGGER.info(f"Working with {self.cfg.data.n_jets} extra jets")

        # load all datasets and organize them in lists
        self.events_raw, self.events_prepd, self.prep_params = [], [], []
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

            # preprocess data
            data_raw = ensure_onshell(data_raw, self.onshell_list, self.onshell_mass)
            if self.modelname == "GATr":
                data_prepd, prep_params = preprocess_gatr(data_raw, self.pt_min)
            elif self.modelname == "Transformer":
                data_prepd, prep_params = preprocess_tr(data_raw, self.pt_min)

            # collect everything
            self.events_raw.append(data_raw)
            self.events_prepd.append(data_prepd)
            self.prep_params.append(prep_params)

        # pass prep_params to model
        if self.modelname == "GATr":
            self.model.prep_params = self.prep_params

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
        self.train_loader = torch.utils.data.DataLoader(
            dataset=EventDataset([x["trn"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=EventDataset([x["tst"] for x in self.data_prepd], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = torch.utils.data.DataLoader(
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
            self.evaluate_log_prob_single(loaders[key], key)

    def _evaluate_loss_single(self, loader, title):
        # use the same random numbers for all datasets to get comparable results
        gen = torch.Generator().manual_seed(42)

        self.model.eval()
        losses = []
        mses = {f"{n_jets}j": [] for n_jets in self.cfg.data.n_jets}
        LOGGER.info(f"Starting to evaluate model on {title} dataset")
        t0 = time.time()
        for i, data in enumerate(loader):
            loss = 0.0
            for ijet, data_single in enumerate(data):
                x0 = data_single.to(self.device)
                loss_single = self.model.batch_loss(x0, ijet, loss_fn=self.loss)
                loss += loss_single / len(self.cfg.data.n_jets)
                mses[f"{self.cfg.data.n_jets[ijet]}j"].append(loss_single.cpu().item())
            losses.append(loss.cpu().item())
        dt = time.time() - t0
        LOGGER.info(f"Finished evaluating {title} dataset after {dt/60:.2f}min")

        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.loss", np.mean(losses))
            for key, values in mses.items():
                log_mlflow(f"eval.{title}.{key}.mse.{key}", np.mean(values))

    def _evaluate_log_prob_single(self, loader, title):
        raise NotImplementedError

    def _sample_events(self):
        self.model.eval()

        for ijet, n_jets in enumerate(self.cfg.data.n_jets):

            sample = []
            shape = (self.cfg.evaluation.batchsize, self.n_hard_particles + n_jets, 4)
            n_batches = (
                1 + (self.cfg.evaluation.nsamples - 1) // self.cfg.evaluation.batchsize
            )
            LOGGER.info(
                f"Starting to generate {self.cfg.evaluation.nsamples} {n_jets}j events"
            )
            t0 = time.time()
            for i in range(n_batches):
                x_t = self.model.sample(
                    ijet, shape, device=self.device, dtype=self.dtype
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

            if self.modelname == "GATr":
                samples_raw = undo_preprocess_gatr(
                    samples, self.pt_min, self.prep_params[ijet]
                )
            elif self.modelname == "Transformer":
                samples_raw = undo_preprocess_tr(
                    samples, self.pt_min, self.prep_params[ijet]
                )
            self.data_raw[ijet]["gen"] = samples_raw

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
            "model_label": "GATr" if self.modelname == "GATr" else "Transformer",
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
        self.loss = torch.nn.MSELoss()

    def _batch_loss(self, data):

        # average over contributions from different datasets
        loss = 0.0
        mse = []
        for ijet, x0 in enumerate(data):
            x0 = x0.to(self.device)
            loss_single = self.model.batch_loss(x0, ijet, loss_fn=self.loss)
            loss += loss_single / len(self.cfg.data.n_jets)
            mse.append(loss_single.cpu().item())
        assert torch.isfinite(loss).all()

        metrics = {
            f"{n_jets}j.mse": mse[ijet]
            for (ijet, n_jets) in enumerate(self.cfg.data.n_jets)
        }
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{n_jets}j.mse": [] for n_jets in self.cfg.data.n_jets}
        return metrics

    def define_process_specifics(self):
        raise NotImplementedError
