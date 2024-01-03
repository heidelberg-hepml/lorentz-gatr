import numpy as np
import torch

import os, sys, time
from omegaconf import OmegaConf, open_dict
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_experiment import BaseExperiment
from experiments.base_plots import plot_loss
from experiments.amplitudes.wrappers import AmplitudeMLPWrapper, AmplitudeTransformerWrapper, \
     AmplitudeCLSTrWrapper, AmplitudeGATrWrapper, AmplitudeGAPWrapper
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import preprocess_particles, preprocess_amplitude, undo_preprocess_amplitude
from experiments.amplitudes.plots import plot_histograms, plot_delta_histogram
from experiments.misc import get_device, flatten_dict
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

TYPE_TOKEN_DICT = {"aag": [0,0,1,1,0], "aagg": [0,0,1,1,0,0],
                    "zgg": [0,0,1,2,2], "zggg": [0,0,1,2,2,2],
                    "zgggg": [0,0,1,2,2,2,2]}
DATASET_TITLE_DICT = {"aag": r"$gg\to\gamma\gamma g$", "aagg": r"$gg\to\gamma\gamma gg$",
                      "zgg": r"$q\bar q\to Zgg$", "zggg": r"$q\bar q\to Zggg$",
                      "zgggg": r"$q\bar q\to Zgggg$"}
MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP", "CLSTr": "CLSTr", "GAP": "GAP"}
BASELINE_MODELS = ["MLP", "Transformer", "CLSTr"]

class AmplitudeExperiment(BaseExperiment):

    def __init__(self, cfg):
        super().__init__(cfg)

    def init_physics(self):
        # experiment-specific adaptations in cfg
        if self.cfg.data.include_permsym:
            self.type_token = TYPE_TOKEN_DICT[self.cfg.data.dataset]
        else:
            self.type_token = list(range(len(TYPE_TOKEN_DICT[self.cfg.data.dataset])))
            
        n_tokens = max(self.type_token) + 1
        OmegaConf.set_struct(self.cfg, True)
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        with open_dict(self.cfg):
            # specify shape for type_token and MLPs
            if modelname == "GATr":
                self.cfg.model.net.in_s_channels = n_tokens
            elif modelname in ["Transformer", "CLSTr"]:
                self.cfg.model.net.in_channels = 4 + n_tokens
            elif modelname == "GAP":
                self.cfg.model.net.in_mv_channels = len(TYPE_TOKEN_DICT[self.cfg.data.dataset])
            elif modelname == "MLP":
                self.cfg.model.net.in_shape = 4 * len(TYPE_TOKEN_DICT[self.cfg.data.dataset])
            else:
                raise ValueError(f"model {modelname} not implemented")

            # reinsert_type_token
            if modelname == "GATr" and self.cfg.model.reinsert_type_token:
                self.cfg.model.net.reinsert_s_channels = list(range(n_tokens))

            # extra outputs for heteroscedastic loss
            if self.cfg.training.heteroscedastic:
                if modelname == "MLP":
                    self.cfg.model.net.out_shape = 2  
                elif modelname in ["Transformer", "CLSTr"]:
                    self.cfg.model.net.out_channels = 2
                elif modelname in ["GATr", "GAP"]:
                    self.cfg.model.net.out_mv_channels = 2

    def init_data(self):
        LOGGER.info(f"Working with dataset {self.cfg.data.dataset} "\
                    f"and type_token={self.type_token}")

        data_path = os.path.join(self.cfg.data.data_path, f"{self.cfg.data.dataset}.npy")
        assert os.path.exists(data_path)
        data_raw = np.load(data_path)
        if self.cfg.data.subsample is not None:
            assert self.cfg.data.subsample < data_raw.shape[0]
            LOGGER.info(f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}")
            data_raw = data_raw[:self.cfg.data.subsample,:]
        self.particles = data_raw[:,:-1]
        self.particles = self.particles.reshape(self.particles.shape[0], self.particles.shape[1]//4, 4)
        self.amplitudes = data_raw[:,[-1]]
        LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

        # preprocess data
        self.amplitudes_prepd, self.amplitudes_mean, self.amplitudes_std = preprocess_amplitude(self.amplitudes)
        if type(self.model.net).__name__ in BASELINE_MODELS:
            self.particles_prepd, self.particles_mean, self.particles_std = preprocess_particles(self.particles)
        else:
            self.particles_prepd = self.particles / self.particles.std()

    def evaluate(self):
        self.amplitudes_pred_train, self.amplitudes_truth_train, self.amplitudes_pred_train_prepd, self.amplitudes_truth_train_prepd \
                                    = self._evaluate_single(self.train_loader, "train")
        self.amplitudes_pred_test, self.amplitudes_truth_test, self.amplitudes_pred_test_prepd, self.amplitudes_truth_test_prepd \
                                   = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd, amplitudes_pred_prepd = np.zeros((0, 1)), np.zeros((0, 1))
        LOGGER.info(f"### Starting to evaluate model on {title} dataset with {loader.dataset.amplitudes.shape[0]} elements ###")
        with torch.no_grad():
            for x, y in loader:
                y_pred = self.model(x.to(self.device), type_token=self.type_token)
                if self.cfg.training.heteroscedastic:
                    y_pred = y_pred[...,[0]]
                amplitudes_pred_prepd = np.concatenate((amplitudes_pred_prepd,
                                                        y_pred.cpu().float().numpy()), axis=0)
                amplitudes_truth_prepd = np.concatenate((amplitudes_truth_prepd,
                                                         y.cpu().float().numpy()), axis=0)
        assert amplitudes_truth_prepd.shape == amplitudes_pred_prepd.shape \
               and amplitudes_truth_prepd.shape == loader.dataset.amplitudes.shape

        # compute metrics over preprocessed amplitudes 
        mse_prepd = np.mean( (amplitudes_pred_prepd - amplitudes_truth_prepd) **2)
        rmse_prepd = np.mean( ( (amplitudes_pred_prepd - amplitudes_truth_prepd) / amplitudes_truth_prepd) **2)

        # undo preprocessing
        amplitudes_truth = undo_preprocess_amplitude(amplitudes_truth_prepd,
                                                     self.amplitudes_mean, self.amplitudes_std)
        amplitudes_pred = undo_preprocess_amplitude(amplitudes_pred_prepd,
                                                  self.amplitudes_mean, self.amplitudes_std)

        # compute metrics
        mse = np.mean( (amplitudes_truth - amplitudes_pred)**2 )
        delta = (amplitudes_truth - amplitudes_pred) / amplitudes_truth
        rmse = np.mean( delta**2)

        delta_maxs = [.001, .01, .1]
        delta_rates = []
        for delta_max in delta_maxs:
            rate = np.mean( (delta > -delta_max) * (delta < delta_max)) # rate of events with -delta_max < delta < delta_max
            delta_rates.append(rate)
        LOGGER.info(f"rate of events in delta interval on {title} dataset:\t"\
                    f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}")

        if self.cfg.use_mlflow:
            log_dict = {f"eval.mse_{title}_raw": mse_prepd,
                        f"eval.rmse_{title}_raw": rmse_prepd,
                        f"eval.mse_{title}": mse,
                        f"eval.rmse_{title}": rmse}
            for key, value in log_dict.items():
                log_mlflow(key, value)

        return amplitudes_pred, amplitudes_truth, amplitudes_truth_prepd, amplitudes_pred_prepd

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        dataset_title = DATASET_TITLE_DICT[self.cfg.data.dataset]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = f"{model_title}: {dataset_title}"
        LOGGER.info(f"Creating plots in {plot_path}")
        
        if self.cfg.plotting.loss and self.cfg.train:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.train_loss, self.val_loss], self.train_lr,
                      labels=["train loss", "val loss"])

        if self.cfg.plotting.histograms and self.cfg.evaluate:
            out = f"{plot_path}/histograms.pdf"
            with PdfPages(out) as file:
                labels = ["Test", "Train", "Prediction"]

                data = [np.log(self.amplitudes_truth_test), np.log(self.amplitudes_truth_train),
                        np.log(self.amplitudes_pred_test)]
                plot_histograms(file, data, labels, title=title,
                           xlabel=r"$\log A$", logx=False)

        if self.cfg.plotting.delta and self.cfg.evaluate:
            out = f"{plot_path}/delta.pdf"
            with PdfPages(out) as file:
                data_test = (self.amplitudes_pred_test[:,0] - self.amplitudes_truth_test[:,0]) / self.amplitudes_truth_test[:,0]
                data_train = (self.amplitudes_pred_train[:,0] - self.amplitudes_truth_train[:,0]) / self.amplitudes_truth_train[:,0]

                # determine 1% largest amplitudes
                scale = self.amplitudes_truth_test[:,0]
                largest_idx = round(.01 * len(scale) )
                sort_idx = np.argsort(scale, axis=0)
                largest_min = scale[sort_idx][-largest_idx-1]
                largest_mask = (scale > largest_min)

                xranges = [(-10.,10.), (-30., 30.), (-100., 100.)] # in %
                binss = [100, 50, 50]
                for xrange, bins in zip(xranges, binss):
                    plot_delta_histogram(file, [data_test*100, data_train*100],
                                         labels=["Test", "Train"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                    plot_delta_histogram(file, [data_test*100, data_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                    plot_delta_histogram(file, [data_test*100, data_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=True)
                    
        if self.cfg.plotting.delta_prepd and self.cfg.evaluate:
            out = f"{plot_path}/delta_prepd.pdf"
            with PdfPages(out) as file:
                data_test = (self.amplitudes_pred_test_prepd[:,0] - self.amplitudes_truth_test_prepd[:,0]) / self.amplitudes_truth_test_prepd[:,0]
                data_train = (self.amplitudes_pred_train_prepd[:,0] - self.amplitudes_truth_train_prepd[:,0]) / self.amplitudes_truth_train_prepd[:,0]

                # determine 1% largest amplitudes
                scale = self.amplitudes_truth_test_prepd[:,0]
                largest_idx = round(.01 * len(scale) )
                sort_idx = np.argsort(scale, axis=0)
                largest_min = scale[sort_idx][-largest_idx-1]
                largest_mask = (scale > largest_min)

                xranges = [(-10.,10.), (-30., 30.), (-100., 100.)] # in %
                binss = [100, 50, 50]
                for xrange, bins in zip(xranges, binss):
                    plot_delta_histogram(file, [data_test*100, data_train*100],
                                         labels=["Test", "Train"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                    plot_delta_histogram(file, [data_test*100, data_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                    plot_delta_histogram(file, [data_test*100, data_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title, 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=True)

    def _init_dataloader(self):
        n_data = self.particles.shape[0]
        assert sum(self.cfg.training.train_test_val) <= 1
        self.split_train = int(n_data * self.cfg.training.train_test_val[0])
        self.split_test = int(n_data * sum(self.cfg.training.train_test_val[:2]))
        self.split_val = int(n_data * sum(self.cfg.training.train_test_val))

        train_dataset = AmplitudeDataset(self.particles_prepd[:self.split_train], self.amplitudes_prepd[:self.split_train],
                                         dtype=self.dtype)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.cfg.training.batchsize, shuffle=True)

        test_dataset = AmplitudeDataset(self.particles_prepd[self.split_train:self.split_test], self.amplitudes_prepd[self.split_train:self.split_test],
                                        dtype=self.dtype)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=self.cfg.evaluation.batchsize, shuffle=False)

        val_dataset = AmplitudeDataset(self.particles_prepd[self.split_test:self.split_val], self.amplitudes_prepd[self.split_test:self.split_val],
                                        dtype=self.dtype)
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                        batch_size=self.cfg.training.batchsize, shuffle=False)
        LOGGER.debug(f"Constructed dataloaders with train_test_val={self.cfg.training.train_test_val}, "\
                     f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "\
                     f"batch_size={self.cfg.training.batchsize}")

    def _init_loss(self):
        if self.cfg.training.heteroscedastic:
            def heteroscedastic_loss(y_true, pred):
                # extract log(sigma^2) instead of just sigma to improve numerical stability
                y_pred, logsigma2 = pred[...,[0]], pred[...,[1]]
                
                # drop constant term log(2 pi)/2 because it does not affect optimization
                expression = (y_pred - y_true)**2 / (2 * logsigma2.exp()) + logsigma2 / 2
                return expression.mean()

            self.loss = heteroscedastic_loss
        else:
            self.loss = torch.nn.MSELoss()

    def _batch_loss(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        y_pred = self.model(x, type_token=self.type_token)
        loss = self.loss(y, y_pred)
        assert torch.isfinite(loss).all()

        mse = torch.mean( (y_pred-y)**2)
        rmse = torch.mean( (y_pred/y - 1)**2)
        metrics = {"mse": mse, "rmse": rmse}
        return loss, metrics

    def _init_metrics(self):
        metrics = {"mse": [], "rmse": []}
        return metrics
