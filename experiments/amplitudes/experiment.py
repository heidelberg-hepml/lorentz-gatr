import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_experiment import BaseExperiment
from experiments.base_plots import plot_loss
from experiments.amplitudes.wrappers import AmplitudeMLPWrapper, AmplitudeTransformerWrapper, \
     AmplitudeGATrWrapper, AmplitudeGAPWrapper
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import preprocess_particles, preprocess_amplitude, undo_preprocess_amplitude
from experiments.amplitudes.plots import plot_histograms, plot_delta_histogram, plot_pull
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

TYPE_TOKEN_DICT = {"aag": [0,0,1,1,0], "aagg": [0,0,1,1,0,0],
                    "zgg": [0,0,1,2,2], "zggg": [0,0,1,2,2,2],
                    "zgggg": [0,0,1,2,2,2,2]}
DATASET_TITLE_DICT = {"aag": r"$gg\to\gamma\gamma g$", "aagg": r"$gg\to\gamma\gamma gg$",
                      "zgg": r"$q\bar q\to Zgg$", "zggg": r"$q\bar q\to Zggg$",
                      "zgggg": r"$q\bar q\to Zgggg$"}
MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP", "GAP": "GAP"}
BASELINE_MODELS = ["MLP", "Transformer"]

class AmplitudeExperiment(BaseExperiment):

    def init_physics(self):
        self.n_datasets = len(self.cfg.data.dataset)
        
        # create type_token list
        self.type_token = []
        for dataset in self.cfg.data.dataset:
            if self.cfg.data.include_permsym:
                self.type_token.append(TYPE_TOKEN_DICT[dataset])
            else:
                self.type_token.append(list(range(len(TYPE_TOKEN_DICT[dataset]))))
            
        n_tokens = max([max(token) for token in self.type_token]) + 1
        OmegaConf.set_struct(self.cfg, True)
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        if modelname in ["GAP", "MLP"]:
            assert len(self.cfg.data.dataset) == 1, f"Architecture {modelname} can not handle several datasets "\
                   f"as specified in {self.cfg.data.dataset}"
            
        with open_dict(self.cfg):
            # specify shape for type_token and MLPs
            if modelname == "GATr":
                self.cfg.model.net.in_s_channels = n_tokens
            elif modelname == "Transformer":
                self.cfg.model.net.in_channels = 4 + n_tokens
            elif modelname == "GAP":
                self.cfg.model.net.in_mv_channels = len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]])
            elif modelname == "MLP":
                self.cfg.model.net.in_shape = 4 * len(TYPE_TOKEN_DICT[self.cfg.data.dataset[0]])
            else:
                raise ValueError(f"model {modelname} not implemented")

            # reinsert_type_token
            if modelname == "GATr" and self.cfg.model.reinsert_type_token:
                self.cfg.model.net.reinsert_s_channels = list(range(n_tokens))

            # extra outputs for heteroscedastic loss
            if self.cfg.heteroscedastic:
                if modelname == "MLP":
                    self.cfg.model.net.out_shape = 2  
                elif modelname == "Transformer":
                    self.cfg.model.net.out_channels = 2
                elif modelname in ["GATr", "GAP"]:
                    self.cfg.model.net.out_mv_channels = 2

    def init_data(self):
        LOGGER.info(f"Working with dataset {self.cfg.data.dataset} "\
                    f"and type_token={self.type_token}")

        # load all datasets and organize them in lists
        self.particles, self.amplitudes, self.particles_prepd, self.amplitudes_prepd, self.prepd_mean, self.prepd_std \
                        = [], [], [], [], [], []
        for dataset in self.cfg.data.dataset:
            # load data
            data_path = os.path.join(self.cfg.data.data_path, f"{dataset}.npy")
            assert os.path.exists(data_path), f"data_path {data_path} does not exist"
            data_raw = np.load(data_path)
            LOGGER.info(f"Loaded data with shape {data_raw.shape} from {data_path}")

            # bring data into correct shape
            if self.cfg.data.subsample is not None:
                assert self.cfg.data.subsample < data_raw.shape[0]
                LOGGER.info(f"Reducing the size of the dataset from {data_raw.shape[0]} to {self.cfg.data.subsample}")
                data_raw = data_raw[:self.cfg.data.subsample,:]
            particles = data_raw[:,:-1]
            particles = particles.reshape(particles.shape[0], particles.shape[1]//4, 4)
            amplitudes = data_raw[:,[-1]]    

            # preprocess data
            amplitudes_prepd, prepd_mean, prepd_std = preprocess_amplitude(amplitudes)
            if type(self.model.net).__name__ in BASELINE_MODELS:
                particles_prepd, _, _ = preprocess_particles(particles)
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
        train_sets, test_sets, val_sets = {"particles": [], "amplitudes": []}, \
                                          {"particles": [], "amplitudes": []}, \
                                          {"particles": [], "amplitudes": []}
        for idataset in range(self.n_datasets):        
            n_data = self.particles[idataset].shape[0]
            self.split_train = int(n_data * self.cfg.data.train_test_val[0])
            self.split_test = int(n_data * sum(self.cfg.data.train_test_val[:2]))
            self.split_val = int(n_data * sum(self.cfg.data.train_test_val))

            train_sets["particles"].append(self.particles_prepd[idataset][0:self.split_train])
            train_sets["amplitudes"].append(self.amplitudes_prepd[idataset][0:self.split_train])
            
            test_sets["particles"].append(self.particles_prepd[idataset][self.split_train:self.split_test])
            test_sets["amplitudes"].append(self.amplitudes_prepd[idataset][self.split_train:self.split_test])

            val_sets["particles"].append(self.particles_prepd[idataset][self.split_test:self.split_val])
            val_sets["amplitudes"].append(self.amplitudes_prepd[idataset][self.split_test:self.split_val])

        # create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(train_sets["particles"], train_sets["amplitudes"], dtype=self.dtype),
            batch_size=self.cfg.training.batchsize, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(test_sets["particles"], test_sets["amplitudes"], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize, shuffle=False)

        self.val_loader = torch.utils.data.DataLoader(
            dataset=AmplitudeDataset(val_sets["particles"], val_sets["amplitudes"], dtype=self.dtype),
            batch_size=self.cfg.evaluation.batchsize, shuffle=False)

        LOGGER.info(f"Constructed dataloaders with train_test_val={self.cfg.data.train_test_val}, "\
                     f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "\
                     f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)")

    def evaluate(self):
        self.results_train = self._evaluate_single(self.train_loader, "train")
        self.results_test = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        # compute predictions
        # note: shuffle=True or False does not matter, because we take the predictions directly from the dataloader and not from the dataset
        amplitudes_truth_prepd, amplitudes_pred_prepd = [[] for _ in range(self.n_datasets)], \
                                                        [[] for _ in range(self.n_datasets)]
        if self.cfg.heteroscedastic: # also save predicted uncertainties
            std_pred_prepd = [[] for _ in range(self.n_datasets)]
        LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        self.model.eval()
        t0 = time.time()
        with torch.no_grad():
            for data in loader:
                for idataset, data_onedataset in enumerate(data):
                    x, y = data_onedataset
                    pred = self.model(x.to(self.device),
                                      type_token=self.type_token[idataset],
                                      global_token=idataset)
                    y_pred = pred[...,0]
                    if self.cfg.heteroscedastic:
                        std_prepd = torch.exp(pred[...,1] / 2) # extract sigma from log(sigma^2)
                        std_pred_prepd[idataset].append(std_prepd.cpu().float().numpy())
                    
                    amplitudes_pred_prepd[idataset].append(y_pred.cpu().float().numpy())
                    amplitudes_truth_prepd[idataset].append(y.flatten().cpu().float().numpy())
        amplitudes_pred_prepd = [np.concatenate(individual) for individual in amplitudes_pred_prepd]
        amplitudes_truth_prepd = [np.concatenate(individual) for individual in amplitudes_truth_prepd]
        if self.cfg.heteroscedastic:
            std_pred_prepd = [np.concatenate(individual) for individual in std_pred_prepd]
        dt = (time.time() - t0) * 1e6/sum(arr.shape[0] for arr in amplitudes_truth_prepd)
        LOGGER.info(f"Evaluation time: {dt:.2f}s for 1M events "\
                    f"using batchsize {self.cfg.evaluation.batchsize}")

        results = {}
        for idataset, dataset in enumerate(self.cfg.data.dataset):
            print(f"STARTING {idataset}")
            amp_pred_prepd = amplitudes_pred_prepd[idataset]
            amp_truth_prepd = amplitudes_truth_prepd[idataset]
            
            # compute metrics over preprocessed amplitudes 
            mse_prepd = np.mean( (amp_pred_prepd - amp_truth_prepd) **2)

            # undo preprocessing
            amp_truth = undo_preprocess_amplitude(amp_truth_prepd,
                                                     self.prepd_mean[idataset], self.prepd_std[idataset])
            amp_pred = undo_preprocess_amplitude(amp_pred_prepd,
                                                  self.prepd_mean[idataset], self.prepd_std[idataset])

            # compute metrics over actual amplitudes
            mse = np.mean( (amp_truth - amp_pred)**2 )
            
            delta = (amp_truth - amp_pred) / amp_truth
            delta_maxs = [.001, .01, .1]
            delta_rates = []
            for delta_max in delta_maxs:
                rate = np.mean( (delta > -delta_max) * (delta < delta_max)) # fraction of events with -delta_max < delta < delta_max
                delta_rates.append(rate)
            LOGGER.info(f"rate of events in delta interval on {dataset} {title} dataset:\t"\
                    f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}")

            # compute pulls
            if self.cfg.heteroscedastic:
                std_pred = std_pred_prepd[idataset] * self.prepd_std[idataset] * amp_pred
            
                pull_prepd = (amp_pred_prepd - amp_truth_prepd) / std_pred_prepd[idataset]
                pull = (amp_pred - amp_truth) / std_pred
            else:
                pull_prepd, pull=None, None

            # log to mlflow
            if self.cfg.use_mlflow:
                log_dict = {f"eval.{title}.{dataset}.mse": mse_prepd,
                        f"eval.{title}.{dataset}.mse_raw": mse}
                if self.cfg.heteroscedastic:
                    log_dict[f"eval.{title}.{dataset}.pull_mean"] = np.mean(pull_prepd)
                    log_dict[f"eval.{title}.{dataset}.pull_std"] = np.std(pull_prepd)
                    log_dict[f"eval.{title}.{dataset}.pull_mean_raw"] = np.mean(pull)
                    log_dict[f"eval.{title}.{dataset}.pull_std_raw"] = np.std(pull)
                for key, value in log_dict.items():
                    log_mlflow(key, value)

            amp = {"raw": {"truth": amp_truth, "prediction": amp_pred, "mse": mse,
                       "pull": pull},
               "preprocessed": {"truth": amp_truth_prepd, "prediction": amp_pred_prepd,
                                "mse": mse_prepd, "pull": pull_prepd}}
            results[dataset] = amp
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        dataset_titles = [DATASET_TITLE_DICT[dataset] for dataset in self.cfg.data.dataset]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = [f"{model_title}: {dataset_title}" for dataset_title in dataset_titles]
        LOGGER.info(f"Creating plots in {plot_path}")
        
        if self.cfg.plotting.loss and self.cfg.train:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.train_loss, self.val_loss], self.train_lr,
                      labels=["train loss", "val loss"],
                      logy=False if self.cfg.heteroscedastic else True) # loss can become negative when heteroscedastic

        if self.cfg.plotting.histograms and self.cfg.evaluate:
            out = f"{plot_path}/histograms.pdf"
            with PdfPages(out) as file:
                labels = ["Test", "Train", "Prediction"]

                for idataset, dataset in enumerate(self.cfg.data.dataset):
                    data = [np.log(self.results_test[dataset]["raw"]["truth"]),
                        np.log(self.results_train[dataset]["raw"]["truth"]),
                        np.log(self.results_test[dataset]["raw"]["prediction"])]
                    plot_histograms(file, data, labels, title=title[idataset],
                           xlabel=r"$\log A$", logx=False)

        if self.cfg.plotting.delta and self.cfg.evaluate:
            out = f"{plot_path}/delta.pdf"
            with PdfPages(out) as file:
                for idataset, dataset in enumerate(self.cfg.data.dataset):
                    delta_test = (self.results_test[dataset]["raw"]["prediction"] - self.results_test[dataset]["raw"]["truth"]) / self.results_test[dataset]["raw"]["truth"]
                    delta_train = (self.results_train[dataset]["raw"]["prediction"] - self.results_train[dataset]["raw"]["truth"]) / self.results_train[dataset]["raw"]["truth"]

                    # determine 1% largest amplitudes
                    scale = self.results_test[dataset]["raw"]["truth"]
                    largest_idx = round(.01 * len(scale) )
                    sort_idx = np.argsort(scale)
                    largest_min = scale[sort_idx][-largest_idx-1]
                    largest_mask = (scale > largest_min)

                    xranges = [(-10.,10.), (-30., 30.), (-100., 100.)] # in %
                    binss = [100, 50, 50]
                    for xrange, bins in zip(xranges, binss):
                        plot_delta_histogram(file, [delta_test*100, delta_train*100],
                                         labels=["Test", "Train"], title=title[idataset], 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                        plot_delta_histogram(file, [delta_test*100, delta_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title[idataset], 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                        plot_delta_histogram(file, [delta_test*100, delta_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title[idataset], 
                                         xlabel=r"$\Delta = \frac{A_\mathrm{pred} - A_\mathrm{true}}{A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=True)
                    
        if self.cfg.plotting.delta_prepd and self.cfg.evaluate:
            out = f"{plot_path}/delta_prepd.pdf"
            with PdfPages(out) as file:
                for idataset, dataset in enumerate(self.cfg.data.dataset):
                    delta_test = (self.results_test[dataset]["preprocessed"]["prediction"] - self.results_test[dataset]["preprocessed"]["truth"]) / self.results_test[dataset]["preprocessed"]["truth"]
                    delta_train = (self.results_train[dataset]["preprocessed"]["prediction"] - self.results_train[dataset]["preprocessed"]["truth"]) / self.results_train[dataset]["preprocessed"]["truth"]

                    # determine 1% largest amplitudes
                    scale = self.results_test[dataset]["preprocessed"]["truth"]
                    largest_idx = round(.01 * len(scale) )
                    sort_idx = np.argsort(scale)
                    largest_min = scale[sort_idx][-largest_idx-1]
                    largest_mask = (scale > largest_min)

                    xranges = [(-10.,10.), (-30., 30.), (-100., 100.)] # in %
                    binss = [100, 50, 50]
                    for xrange, bins in zip(xranges, binss):
                        plot_delta_histogram(file, [delta_test*100, delta_train*100],
                                         labels=["Test", "Train"], title=title[idataset], 
                                         xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                        plot_delta_histogram(file, [delta_test*100, delta_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title[idataset], 
                                         xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=False)
                        plot_delta_histogram(file, [delta_test*100, delta_test[largest_mask]*100],
                                         labels=["Test", "Largest 1\%"], title=title[idataset], 
                                         xlabel=r"$\tilde\Delta = \frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde A_\mathrm{true}}$ [\%]",
                                         xrange=xrange, bins=bins, logy=True)

        if self.cfg.plotting.pull and self.cfg.evaluate and self.cfg.heteroscedastic:
            out = f"{plot_path}/pull.pdf"
            with PdfPages(out) as file:
                for idataset, dataset in enumerate(self.cfg.data.dataset):
                    pulls = [self.results_train[dataset]["preprocessed"]["pull"], self.results_test[dataset]["preprocessed"]["pull"]]
                    plot_pull(file, pulls, ["Train", "Test"], r"$\frac{\tilde A_\mathrm{pred} - \tilde A_\mathrm{true}}{\tilde \sigma_\mathrm{pred}}$",
                          title=title[idataset], xrange=(-5,5), bins=60, logy=False)
                
                    pulls = [self.results_train[dataset]["raw"]["pull"], self.results_test[dataset]["raw"]["pull"]]
                    plot_pull(file, pulls, ["Train", "Test"], r"$\frac{A_\mathrm{pred} - A_\mathrm{true}}{\sigma_\mathrm{pred}}$",
                          title=title[idataset], xrange=(-5,5), bins=60, logy=False)

    def _init_loss(self):
        if self.cfg.heteroscedastic:
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
        # average over contributions from different datasets
        loss = torch.empty(self.n_datasets)
        mse = []
        for idataset, data_onedataset in enumerate(data):
            x, y = data_onedataset
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x,
                                type_token=self.type_token[idataset],
                                global_token=idataset)
            loss[idataset] = self.loss(y, y_pred)
            mse.append(torch.mean( (y_pred[:,0]-y[:,0])**2).cpu().item())
        loss = loss.mean()
        assert torch.isfinite(loss).all()

        metrics = {f"{dataset}.mse": mse[i] for (i, dataset) in enumerate(self.cfg.data.dataset)}
        return loss, metrics

    def _init_metrics(self):
        metrics = {f"{dataset}.mse": [] for dataset in self.cfg.data.dataset}
        return metrics
