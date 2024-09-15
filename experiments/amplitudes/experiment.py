import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict

from experiments.base_experiment import BaseExperiment
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import (
    preprocess_particles,
    preprocess_particles_w_invariants,
    preprocess_amplitude,
    undo_preprocess_amplitude,
)
from experiments.amplitudes.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

TYPE_TOKEN_DICT = {
    "aag": [0, 0, 1, 1, 0],
    "aagg": [0, 0, 1, 1, 0, 0],
    "zg": [0, 0, 1, 2],
    "zgg": [0, 0, 1, 2, 2],
    "zggg": [0, 0, 1, 2, 2, 2],
    "zgggg": [0, 0, 1, 2, 2, 2, 2],
    "zggggg": [0, 0, 1, 2, 2, 2, 2, 2],
}
DATASET_TITLE_DICT = {
    "aag": r"$gg\to\gamma\gamma g$",
    "aagg": r"$gg\to\gamma\gamma gg$",
    "zg": r"$q\bar q\to Zg$",
    "zgg": r"$q\bar q\to Zgg$",
    "zggg": r"$q\bar q\to Zggg$",
    "zgggg": r"$q\bar q\to Zgggg$",
    "zggggg": r"$q\bar q \to Zggggg$",
}
MODEL_TITLE_DICT = {
    "GATr": "GATr",
    "Transformer": "Tr",
    "MLP": "MLP",
    "GAP": "GAP",
    "DSI": "DSI",
}
BASELINE_MODELS = ["MLP", "Transformer"]
VB_MODELS = ["DSI"]


class AmplitudeExperiment(BaseExperiment):
    def init_physics(self):
        assert (
            not self.cfg.training.force_xformers
        ), "amplitudes experiment assumes default torch attention"
        self.n_datasets = len(self.cfg.data.dataset)

        # create type_token list
        self.type_token = []
        for dataset in self.cfg.data.dataset:
            if self.cfg.data.include_permsym:
                self.type_token.append(TYPE_TOKEN_DICT[dataset])
            else:
                self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))

        token_size = max(
            [max([max(token) for token in self.type_token]) + 1, self.n_datasets]
        )
        OmegaConf.set_struct(self.cfg, True)
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        if modelname in ["GAP", "MLP", "DSI"]:
            assert len(self.cfg.data.dataset) == 1, (
                f"Architecture {modelname} can not handle several datasets "
                f"as specified in {self.cfg.data.dataset}"
            )

        with open_dict(self.cfg):
            # specify shape for type_token and MLPs
            if modelname == "GATr":
                self.cfg.model.net.in_s_channels = token_size
                self.cfg.model.token_size = token_size
            elif modelname == "Transformer":
                self.cfg.model.net.in_channels = 4 + token_size
                self.cfg.model.token_size = token_size
            elif modelname == "GAP":
                self.cfg.model.net.in_mv_channels = len(
                    TYPE_TOKEN_DICT[self.cfg.data.dataset[0]]
                )
            elif modelname == "MLP":
                self.cfg.model.net.in_shape = 4 * len(
                    TYPE_TOKEN_DICT[self.cfg.data.dataset[0]]
                )
            elif modelname == "DSI":
                self.cfg.model.net.in_shape = (
                    4 * len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]])
                    + (len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]]) - 1)
                    * (len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]]))
                    // 2
                )
                self.cfg.model.net.num_particles_boson = TYPE_TOKEN_DICT[
                    self.cfg.data.dataset[0]
                ][2:].count(1)
                self.cfg.model.net.num_particles_glu = TYPE_TOKEN_DICT[
                    self.cfg.data.dataset[0]
                ][2:].count(0) + TYPE_TOKEN_DICT[self.cfg.data.dataset[0]][2:].count(2)
            else:
                raise ValueError(f"model {modelname} not implemented")

    def init_data(self):
        LOGGER.info(
            f"Working with dataset {self.cfg.data.dataset} "
            f"and type_token={self.type_token}"
        )

        # load all datasets and organize them in lists
        (
            self.particles,
            self.amplitudes,
            self.particles_prepd,
            self.amplitudes_prepd,
            self.prepd_mean,
            self.prepd_std,
        ) = ([], [], [], [], [], [])
        for dataset in self.cfg.data.dataset:
            # load data
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
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
            particles = data_raw[:, :-1]
            particles = particles.reshape(
                particles.shape[0], particles.shape[1] // 4, 4
            )
            amplitudes = data_raw[:, [-1]]

            # preprocess data
            amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(amplitudes)
            if type(self.model.net).__name__ in BASELINE_MODELS:
                particles_prepd, _, _ = preprocess_particles(particles)
            elif type(self.model.net).__name__ in VB_MODELS:
                particles_prepd = preprocess_particles_w_invariants(particles)
            else:
                particles_prepd = particles / particles.std()

            # collect everything
            self.particles.append(particles)
            self.amplitudes.append(amplitudes)
            self.particles_prepd.append(particles_prepd)
            self.amplitudes_prepd.append(amplitudes_prepd)
            self.prepd_mean.append(prepd_mean)
            self.prepd_std.append(prepd_std)

    def _init_dataloader(self):
        assert sum(self.cfg.data.train_test_val) <= 1

        # seperate data into train, test and validation subsets for each dataset
        train_sets, test_sets, val_sets = (
            {"particles": [], "amplitudes": []},
            {"particles": [], "amplitudes": []},
            {"particles": [], "amplitudes": []},
        )
        for idataset in range(self.n_datasets):
            n_data = self.particles[idataset].shape[0]
            self.split_train = int(n_data * self.cfg.data.train_test_val[0])
            self.split_test = int(n_data * sum(self.cfg.data.train_test_val[:2]))
            self.split_val = int(n_data * sum(self.cfg.data.train_test_val))

            train_sets["particles"].append(
                self.particles_prepd[idataset][0 : self.split_train]
            )
            train_sets["amplitudes"].append(
                self.amplitudes_prepd[idataset][0 : self.split_train]
            )

            test_sets["particles"].append(
                self.particles_prepd[idataset][self.split_train : self.split_test]
            )
            test_sets["amplitudes"].append(
                self.amplitudes_prepd[idataset][self.split_train : self.split_test]
            )

            val_sets["particles"].append(
                self.particles_prepd[idataset][self.split_test : self.split_val]
            )
            val_sets["amplitudes"].append(
                self.amplitudes_prepd[idataset][self.split_test : self.split_val]
            )

        # create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                train_sets["particles"], train_sets["amplitudes"], dtype=self.dtype
            ),
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                test_sets["particles"], test_sets["amplitudes"], dtype=self.dtype
            ),
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        self.val_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(
                val_sets["particles"], val_sets["amplitudes"], dtype=self.dtype
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
            if self.ema is not None:
                with self.ema.average_parameters():
                    self.results_train = self._evaluate_single(
                        self.train_loader, "train"
                    )
                    self.results_val = self._evaluate_single(self.val_loader, "val")
                    self.results_test = self._evaluate_single(self.test_loader, "test")

                # also evaluate without ema to see the effect
                self._evaluate_single(self.train_loader, "train_noema")
                self._evaluate_single(self.val_loader, "val_noema")
                self._evaluate_single(self.test_loader, "test_noema")

            else:
                self.results_train = self._evaluate_single(self.train_loader, "train")
                self.results_val = self._evaluate_single(self.val_loader, "val")
                self.results_test = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd, amplitudes_pred_prepd = [
            [] for _ in range(self.n_datasets)
        ], [[] for _ in range(self.n_datasets)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        t0 = time.time()
        for data in loader:
            for idataset, data_onedataset in enumerate(data):
                x, y = data_onedataset
                x = x.unsqueeze(0)
                pred = self.model(
                    x.to(self.device),
                    type_token=torch.tensor(
                        [self.type_token[idataset]],
                        dtype=torch.long,
                        device=self.device,
                    ),
                    global_token=torch.tensor(
                        [idataset], dtype=torch.long, device=self.device
                    ),
                )

                y_pred = pred[0, ..., 0]

                amplitudes_pred_prepd[idataset].append(y_pred.cpu().float().numpy())
                amplitudes_truth_prepd[idataset].append(
                    y.flatten().cpu().float().numpy()
                )
        amplitudes_pred_prepd = [
            np.concatenate(individual) for individual in amplitudes_pred_prepd
        ]
        amplitudes_truth_prepd = [
            np.concatenate(individual) for individual in amplitudes_truth_prepd
        ]
        dt = (
            (time.time() - t0)
            * 1e6
            / sum(arr.shape[0] for arr in amplitudes_truth_prepd)
        )
        LOGGER.info(
            f"Evaluation time: {dt:.2f}s for 1M events "
            f"using batchsize {self.cfg.evaluation.batchsize}"
        )

        results = {}
        for idataset, dataset in enumerate(self.cfg.data.dataset):
            amp_pred_prepd = amplitudes_pred_prepd[idataset]
            amp_truth_prepd = amplitudes_truth_prepd[idataset]

            # compute metrics over preprocessed amplitudes
            mse_prepd = np.mean((amp_pred_prepd - amp_truth_prepd) ** 2)

            # undo preprocessing
            amp_truth = undo_preprocess_amplitude(
                amp_truth_prepd, self.prepd_mean[idataset], self.prepd_std[idataset]
            )
            amp_pred = undo_preprocess_amplitude(
                amp_pred_prepd, self.prepd_mean[idataset], self.prepd_std[idataset]
            )

            # compute metrics over actual amplitudes
            mse = np.mean((amp_truth - amp_pred) ** 2)

            delta = (amp_truth - amp_pred) / amp_truth
            delta_maxs = [0.001, 0.01, 0.1]
            delta_rates = []
            for delta_max in delta_maxs:
                rate = np.mean(
                    (delta > -delta_max) * (delta < delta_max)
                )  # fraction of events with -delta_max < delta < delta_max
                delta_rates.append(rate)
            LOGGER.info(
                f"rate of events in delta interval on {dataset} {title} dataset:\t"
                f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}"
            )

            # log to mlflow
            if self.cfg.use_mlflow:
                log_dict = {
                    f"eval.{title}.{dataset}.mse": mse_prepd,
                    f"eval.{title}.{dataset}.mse_raw": mse,
                }
                for key, value in log_dict.items():
                    log_mlflow(key, value)

            amp = {
                "raw": {
                    "truth": amp_truth,
                    "prediction": amp_pred,
                    "mse": mse,
                },
                "preprocessed": {
                    "truth": amp_truth_prepd,
                    "prediction": amp_pred_prepd,
                    "mse": mse_prepd,
                },
            }
            results[dataset] = amp
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        dataset_titles = [
            DATASET_TITLE_DICT[dataset] for dataset in self.cfg.data.dataset
        ]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict["results_test"] = self.results_test
            plot_dict["results_train"] = self.results_train
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.MSELoss()

    def _batch_loss(self, data):
        # average over contributions from different datasets
        loss = 0.0
        mse = []
        if len(data) == 1:
            x, y = data[0]
            x, y = x.unsqueeze(0), y.unsqueeze(0)
            attn_mask = None
            type_token = torch.tensor(
                self.type_token, dtype=torch.long, device=self.device
            )
            global_token = torch.tensor([0], dtype=torch.long, device=self.device)
        else:
            particles_max = data[-1][0].shape[-2]
            assert particles_max == max([d[0].shape[-2] for d in data])
            y = torch.stack([d[1] for d in data], dim=0)
            x = torch.zeros(len(data), *data[-1][0].shape, dtype=self.dtype)
            # carefully construct padding attention mask as float (and not bool!)
            # bool padding mask does not work, because a full row/column of zeros yields nan's in the softmax
            # this will hopefully be fixed soon, see https://github.com/pytorch/pytorch/issues/103749
            attn_mask = (
                torch.ones(
                    len(data),
                    1,
                    1,
                    1 + particles_max,
                    1 + particles_max,
                    dtype=self.dtype,
                )
                * torch.finfo(self.dtype).min
            )
            type_token = torch.zeros(
                len(data), particles_max, dtype=torch.long, device=self.device
            )
            for i, d in enumerate(data):
                particles_i = d[0].shape[-2]
                x[i, :, :particles_i, :] = d[0]
                attn_mask[i, :, :, : (1 + particles_i), : (1 + particles_i)] = 0.0
                type_token[i, :particles_i] = torch.tensor(
                    self.type_token[i], dtype=torch.long, device=self.device
                )
            global_token = torch.tensor(
                range(len(data)), dtype=torch.long, device=self.device
            )
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.model(
            x, type_token=type_token, global_token=global_token, attn_mask=attn_mask
        )
        loss = self.loss(y, y_pred)
        mse = [
            torch.mean((y_pred[i, :, 0] - y[i, :, 0]) ** 2).cpu().item()
            for i in range(len(data))
        ]
        assert torch.isfinite(loss).all()

        metrics = {
            f"{dataset}.mse": mse[i]
            for (i, dataset) in enumerate(self.cfg.data.dataset)
        }
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{dataset}.mse": [] for dataset in self.cfg.data.dataset}
        return metrics
