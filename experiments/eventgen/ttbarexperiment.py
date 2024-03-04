import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict
from torchdiffeq import odeint

from experiments.base_experiment import BaseExperiment
from experiments.eventgen.dataset import EventDataset
from experiments.eventgen.preprocessing import preprocess1, undo_preprocess1
#from experiments.eventgen.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

class ttbarExperiment(BaseExperiment):
    def init_physics(self):
        # define experiment properties
        self.plot_title = r"t\bar t"
        self.n_hard_particles = 6
        self.n_jets_max = 4
        self.obs_names_index = ["b1", "q1", "q2", "b2", "q3", "q4"]
        for ijet in range(self.n_jets_max):
            self.obs_names_index.append(f"j{ijet+1}")
        self.fourmomentum_ranges = [[0, 200], [-150, 150], [-150, 150], [-150, 150]]
        self.jetmomentum_ranges = [[10, 150], [-np.pi, np.pi], [-6,6], [0, 20]]
        self.virtual_components = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2],
            [3, 4, 5],
            [1, 2],
            [4, 5],
        ]
        self.virtual_names = [
            r"p_{T,t\bar t}",
            r"\phi_{t\bar t}",
            r"\eta_{t\bar t}",
            r"m_{t\bar t}",
            "p_{T,t}",
            "\phi_t",
            "\eta_t",
            "m_{ t }",
            r"p_{T,\bar t}",
            r"\phi_{\bar t}",
            r"\eta_{\bar t}",
            r"m_{\bar t}",
            "p_{T,W^+}",
            "\phi_{W^+}",
            "\eta_{W^+}",
            "m_{W^+}",
            "p_{T,W^-}",
            "\phi_{W^-}",
            "\eta_{W^-}",
            "m_{W^-}",
        ]
        self.virtual_ranges = [
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [200, 1000],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 500],
            [-np.pi, np.pi],
            [-6, 6],
            [50, 400],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
            [0, 300],
            [-np.pi, np.pi],
            [-6, 6],
            [30, 150],
        ]

        # dynamically set wrapper properties
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        with open_dict(self.cfg):
            self.cfg.model.type_token_channels = self.n_hard_particles + max(self.cfg.data.n_jets)
            self.cfg.model.process_token_channels = len(self.cfg.data.n_jets)
            if modelname == "Transformer":
                self.cfg.model.net.in_channels = 4 + self.cfg.model.type_token_channels + self.cfg.model.process_token_channels + self.cfg.model.embed_t_dim
            else:
                raise ValueError

    def init_data(self):
        LOGGER.info(
            f"Working with {self.cfg.data.n_jets} extra jets"
        )

        # load all datasets and organize them in lists
        self.events, self.events_prepd, self.prep_params = [], [], []
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
            data_raw = data_raw.reshape(data_raw.shape[0], data_raw.shape[1]//4, 4)

            # preprocess data
            data_prepd, prep_params = preprocess1(data_raw)

            # collect everything
            self.events.append(data_raw)
            self.events_prepd.append(data_prepd)
            self.prep_params.append(prep_params)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets for each dataset
        train_sets, test_sets, val_sets = [], [], []
        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            n_data = self.events[ijet].shape[0]
            split_train = int(n_data * self.cfg.data.train_test_val[0])
            split_test = int(n_data * sum(self.cfg.data.train_test_val[:2]))
            split_val = int(n_data * sum(self.cfg.data.train_test_val))

            train_sets.append(self.events[ijet][0:split_train])
            test_sets.append(self.events[ijet][split_train:split_test])
            val_sets.append(self.events[ijet][split_test:split_val])

        # create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset=EventDataset(
                train_sets, dtype=self.dtype
            ),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=EventDataset(
                test_sets, dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = torch.utils.data.DataLoader(
            dataset=EventDataset(
                val_sets, dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def evaluate(self):
        with torch.no_grad():
            self._sample_events()

            for loader, title in zip([self.train_loader, self.test_loader, self.val_loader, self.sample_loader],
                                     ["train", "test", "val", "samples"]):
                # TODO: EMA-support
                self._evaluate_single(loader, title)

    def _evaluate_single(self, loader, title):
        # TODO
        mses = []

    def _sample_events(self):
        self.model.eval()
        
        self.samples_raw, self.samples_prepd = [], []
        for ijet, n_jets in enumerate(self.cfg.data.n_jets):
            def velocity(t, x_t):
                t = t * torch.ones(x_t.shape[0], 1, 1, dtype=x_t.dtype, device=x_t.device)
                v_t = self.model(x_t, t, ijet=ijet)
                return v_t

            sample = []
            n_batches = 1 + self.cfg.evaluation.nsamples // self.cfg.evaluation.batchsize
            LOGGER.info(f"Starting to generate {self.cfg.evaluation.nsamples} {n_jets}j events")
            t0 = time.time()
            for i in range(n_batches):
                epsilon = torch.randn(self.cfg.evaluation.batchsize, self.n_hard_particles + n_jets, 4,
                                      dtype=self.dtype, device=self.device)
                x_t = odeint(velocity, epsilon, torch.tensor([1., 0.]))[-1]
                sample.append(x_t)
            t1 = time.time()
            LOGGER.info(f"Finished generating {n_jets}j events after {t1-t0:.2f}s = {(t1-t0)/60:.2f}min")
            
            sample = torch.cat(sample, dim=0)[:self.cfg.evaluation.nsamples,...].cpu().numpy()
            self.samples_prepd.append(sample)

            samples_raw = undo_preprocess1(sample, self.prep_params[ijet])
            self.samples_raw.append(samples_raw)

        self.sample_loader = torch.utils.data.DataLoader(
            dataset=EventDataset(
                self.samples_prepd, dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

    def plot(self):
        raise NotImplementedError

    def _init_loss(self):
        self.loss = torch.nn.MSELoss()

    def _batch_loss(self, data):
        
        # average over contributions from different datasets
        loss = 0.0
        mse = []
        for ijet, x0 in enumerate(data):
            x0 = x0.to(self.device)
            t = torch.rand(x0.shape[0], 1, 1, dtype=x0.dtype, device=x0.device)
            eps = torch.randn_like(x0)
            x_t = (1-t)*x0 + t*eps
            v_t = -x0 + eps

            v_theta = self.model(x_t, t, ijet=ijet)

            loss += self.loss(v_theta, v_t) / len(self.cfg.data.n_jets)
            mse.append(self.loss(v_theta, v_t).cpu().item())
        assert torch.isfinite(loss).all()

        metrics = {
            f"{n_jets}j.mse": mse[ijet]
            for (ijet, n_jets) in enumerate(self.cfg.data.n_jets)
        }
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{n_jets}j.mse": [] for n_jets in self.cfg.data.n_jets}
        return metrics
