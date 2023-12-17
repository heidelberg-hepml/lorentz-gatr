import numpy as np
import torch

import os, sys, time
import zipfile
import logging
from pathlib import Path
from omegaconf import OmegaConf, open_dict, errors
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from matplotlib.backends.backend_pdf import PdfPages

from experiments.amplitudes.wrappers import AmplitudeMLPWrapper, AmplitudeTransformerWrapper, \
     AmplitudeCLSTrWrapper, AmplitudeGATrWrapper, AmplitudeGAMLPWrapper
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import preprocess_particles, preprocess_amplitude, undo_preprocess_amplitude
from experiments.amplitudes.plots import plot_histograms, plot_loss, plot_single_histogram
from experiments.misc import get_device
import experiments.logger
from experiments.logger import LOGGER, MEMORY_HANDLER, FORMATTER

from gatr.layers import MLPConfig, SelfAttentionConfig
cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)


TYPE_TOKEN_DICT = {"aag": [0,1,2,2,3], "aagg": [0,1,2,2,3,3],
                    "zjj": [0,1,2,3,3], "zjjj": [0,1,2,3,3,3],
                    "zjjjj": [0,1,2,3,3,3,3]}
DATASET_TITLE_DICT = {"aag": r"$gg\to\gamma\gamma g$", "aagg": r"$gg\to\gamma\gamma gg$",
                      "zjj": r"$q\bar q\to Zjj$", "zjjj": r"$q\bar q\to Zjjj$",
                      "zjjjj": r"$q\bar q\to Zjjjj$"}
MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP", "CLSTr": "CLSTr", "GAMLP": "GAMLP"}
BASELINE_MODELS = ["MLP", "Transformer", "CLSTr"]

class AmplitudeExperiment:

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        # pass all exceptions to the logger
        try:
            self.full_run()
        except errors.ConfigAttributeError:
            LOGGER.exception("Tried to access key that is not specified in the config files")
        except:
            LOGGER.exception("Exiting with error")

        # print buffered logger messages if failed
        if not experiments.logger.logging_initialized:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            MEMORY_HANDLER.setTarget(stream_handler)
            MEMORY_HANDLER.close()

            # this only prints stdout, not the stack trace :(
                
    def full_run(self):
        # implement all ml boilerplate as private methods (_name)
        t0 = time.time()
        
        # initialize environment
        self._init_experiment()
        self._init_directory()
        self._init_logger()
        self._init_backend()

        self.init_physics()
        self.init_model()
        self.init_data()
        self._init_dataloader()

        if self.cfg.train:
            self._init_optimizer()
            self._init_scheduler()
            self.train()
            self._save_model()

        if self.cfg.evaluate:
            self.evaluate()

        if self.cfg.plot:
            self.plot()

        dt = time.time() - t0
        LOGGER.info(f"Finished experiment after {dt/60:.2f}min = {dt/60**2:.2f}h")

    def init_physics(self):
        # experiment-specific adaptations in cfg
        if self.cfg.data.include_permsym:
            self.type_token = TYPE_TOKEN_DICT[self.cfg.data.dataset]
        else:
            self.type_token = list(range(len(TYPE_TOKEN_DICT[self.cfg.data.dataset])))
        n_tokens = np.unique(self.type_token).shape[0]
        OmegaConf.set_struct(self.cfg, True)
        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        with open_dict(self.cfg):
            # specify shape of type_token
            if modelname == "GATr" or modelname == "CLSGATr":
                self.cfg.model.net.in_s_channels = n_tokens
            elif modelname == "Transformer" or modelname == "CLSTr":
                self.cfg.model.net.in_channels = 4 + n_tokens
            elif modelname == "GAMLP":
                self.cfg.model.net.in_mv_channels = len(TYPE_TOKEN_DICT[self.cfg.data.dataset])
            elif modelname == "MLP":
                self.cfg.model.net.in_shape = 4 * len(TYPE_TOKEN_DICT[self.cfg.data.dataset])
            else:
                raise ValueError(f"model {modelname} not implemented")

    def init_model(self):
        # initialize model
        self.model = instantiate(self.cfg.model) # hydra magic
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        LOGGER.info(f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters")

        # load existing model if specified
        if self.warm_start:
            model_path = os.path.join(self.cfg.exp_dir, "models", f"model_{self.cfg.warm_start_idx}.pt")
            try:
                state_dict = torch.load(model_path, map_location="cpu")
            except FileNotFoundError:
                raise ValueError(f"Cannot load model from {model_path}")
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)

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
        LOGGER.info(f"Starting to evaluate model on {title} dataset with {loader.dataset.amplitudes.shape[0]} elements")
        with torch.no_grad():
            for x, y in loader:
                y_pred = self.model(x.to(self.device), type_token=self.type_token)
                amplitudes_pred_prepd = np.concatenate((amplitudes_pred_prepd,
                                                        y_pred.cpu().numpy()), axis=0)
                amplitudes_truth_prepd = np.concatenate((amplitudes_truth_prepd,
                                                         y.cpu().numpy()), axis=0)
        assert amplitudes_truth_prepd.shape == amplitudes_pred_prepd.shape \
               and amplitudes_truth_prepd.shape == loader.dataset.amplitudes.shape

        # compute metrics over preprocessed amplitudes 
        mse_prepd = np.mean( (amplitudes_truth_prepd - amplitudes_pred_prepd) **2)
        mae_prepd = np.linalg.norm(amplitudes_truth_prepd - amplitudes_pred_prepd, axis=1).mean()
        LOGGER.info(f"logA_pred - logA_truth on {title} dataset:\t\tMSE={mse_prepd:.4e}, MAE={mae_prepd:.4e}")

        # undo preprocessing
        amplitudes_truth = undo_preprocess_amplitude(amplitudes_truth_prepd,
                                                     self.amplitudes_mean, self.amplitudes_std)
        amplitudes_pred = undo_preprocess_amplitude(amplitudes_pred_prepd,
                                                  self.amplitudes_mean, self.amplitudes_std)

        # compute metrics
        delta = (amplitudes_truth - amplitudes_pred) / amplitudes_truth
        LOGGER.info(f"delta_mean = {np.mean(delta):.4f}, delta_std = {np.std(delta):.4f}")
        mse = np.mean( delta**2)
        mae = np.linalg.norm(delta, axis=1).mean()
        LOGGER.info(f"delta=(A_pred-A_truth)/A_truth on {title} dataset:\tMSE={mse:.4e}, MAE={mae:.4e}")

        delta_maxs = [.001, .01, .1]
        delta_rates = []
        for delta_max in delta_maxs:
            rate = np.mean( (delta > -delta_max) * (delta < delta_max)) # rate of events with -delta_max < delta < delta_max
            delta_rates.append(rate)
        LOGGER.info(f"rate of events in delta interval on {title} dataset:\t"\
                    f"{[f'{delta_rates[i]:.4f} ({delta_maxs[i]})' for i in range(len(delta_maxs))]}")

        return amplitudes_pred, amplitudes_truth, amplitudes_truth_prepd, amplitudes_pred_prepd

    def plot(self):
        plot_path = os.path.join(self.cfg.exp_dir, f"plots_{self.cfg.exp_idx}")
        os.makedirs(plot_path)
        dataset_title = DATASET_TITLE_DICT[self.cfg.data.dataset]
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = f"{model_title}: {dataset_title}"
        
        if self.cfg.plotting.loss and self.cfg.train:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.metrics["loss"]], self.metrics["lr"], labels=["loss"])

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
                data_test = (self.amplitudes_truth_test - self.amplitudes_pred_test) / self.amplitudes_truth_test
                data_train = (self.amplitudes_truth_train - self.amplitudes_pred_train) / self.amplitudes_truth_train

                plot_single_histogram(file, data_test*100, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ [\%]",
                           logx=False, xrange=(-10, 10), bins=200)
                plot_single_histogram(file, data_train*100, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ [\%] (train)",
                           logx=False, xrange=(-10, 10), bins=200)
                
                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-.3, .3), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-.3, .3), bins=50)

                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-1., 1.), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-1., 1.), bins=50)

                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-10., 10.), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-10., 10.), bins=50)

        if self.cfg.plotting.delta_prepd and self.cfg.evaluate:
            out = f"{plot_path}/delta_prepd.pdf"
            with PdfPages(out) as file:
                data_test = (self.amplitudes_truth_test_prepd - self.amplitudes_pred_test_prepd) / self.amplitudes_truth_test_prepd
                data_train = (self.amplitudes_truth_train_prepd - self.amplitudes_pred_train_prepd) / self.amplitudes_truth_train_prepd

                plot_single_histogram(file, data_test*100, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ [\%]",
                           logx=False, xrange=(-10, 10), bins=200)
                plot_single_histogram(file, data_train*100, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ [\%] (train)",
                           logx=False, xrange=(-10, 10), bins=200)
                
                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-.3, .3), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-.3, .3), bins=50)

                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-1., 1.), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-1., 1.), bins=50)

                plot_single_histogram(file, data_test, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-10., 10.), bins=50)
                plot_single_histogram(file, data_train, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$ (train)",
                           logx=False, xrange=(-10., 10.), bins=50)
    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_idx is None else True

        modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
        with open_dict(self.cfg):
            # append modelname to exp_name
            self.cfg.exp_name = f"{self.cfg.exp_name}_{modelname}"

        if not self.warm_start:
            rnd_number = np.random.randint(low=0, high=9999)
            exp_name = f"{self.cfg.exp_name}_{rnd_number:04}"
            self.exp_dir = os.path.join(self.cfg.base_dir, "runs", exp_name)
            exp_idx = 0
            LOGGER.info(f"Creating new experiment {self.exp_dir}")
            
        else:
            exp_idx = self.cfg.exp_idx + 1
            LOGGER.info(f"Warm-starting from existing experiment {self.cfg.exp_dir} for run {exp_idx}")

        with open_dict(self.cfg):
            self.cfg.exp_idx = exp_idx
            if not self.warm_start:
                self.cfg.warm_start_idx = 0
                self.cfg.exp_dir = self.exp_dir

        LOGGER.debug(OmegaConf.to_yaml(self.cfg))

        # set seed
        if self.cfg.seed is not None:
            LOGGER.info(f"Using seed {self.cfg.seed}")
            torch.random.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

    def _init_directory(self):
        # create experiment directory
        exp_dir = Path(self.cfg.exp_dir).resolve()
        if exp_dir.exists() and not self.warm_start:
            LOGGER.error(f"Experiment in directory {self.cfg.exp_dir} alredy exists. Aborting.")
            exit()
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)

        # save config
        self._save_config("amplitudes.yaml")
        self._save_config(f"amplitudes_{self.cfg.exp_idx}.yaml")

        # save source
        if self.cfg.save_source:
            zip_name = os.path.join(self.cfg.exp_dir, "source.zip")
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            path_gatr = os.path.join(self.cfg.base_dir, "gatr")
            path_experiment = os.path.join(self.cfg.base_dir, "experiments")
            for path in [path_gatr, path_experiment]:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, path))
            zipf.close()
            LOGGER.info(f"Saved source to {zip_name}")

    def _init_logger(self):
        if experiments.logger.logging_initialized:
            LOGGER.info("Logger already initialized")
            return
        
        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        
        # init file_handler
        file_handler = logging.FileHandler(Path(self.cfg.exp_dir) / f"out_{self.cfg.exp_idx}.log")
        file_handler.setFormatter(FORMATTER)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(FORMATTER)

        # flush memory to stream_handler
        # this allows to catch logs that were created before the logger was initialized
        MEMORY_HANDLER.setTarget(stream_handler) # can only flush to one handler, choose stream_handler
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)

        # add new handlers to logger
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)
        LOGGER.propagate = False # avoid duplicate log outputs

        experiments.logger.logging_initialized = True
        LOGGER.info("Logger initialized")

    def _init_backend(self):
        self.device = get_device()
        
        if self.cfg.training.float16 and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            LOGGER.info("Using dtype bfloat16")
        elif self.cfg.training.float16:
            self.dtype = torch.float16
            LOGGER.info("Using dtype float16 (bfloat16 is not supported by environment)")
        else:
            self.dtype = torch.float32
            LOGGER.info("Using dtype float32")

        torch.backends.cuda.enable_flash_sdp(self.cfg.training.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.training.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(self.cfg.training.enable_mem_efficient_sdp)
        if self.cfg.training.force_xformers:
            LOGGER.debug("Forcing use of xformers' attention implementation")
            gatr.primitives.attention.FORCE_XFORMERS = True
        
    def _save_config(self, filename="amplitudes.yaml"):
        # Save config
        config_filename = Path(self.cfg.exp_dir) / filename
        LOGGER.info(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

    def _init_optimizer(self):
        if self.cfg.training.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.cfg.training.lr,
                                              betas=self.cfg.training.betas,
                                              eps=self.cfg.training.eps)
        else:
            raise ValueError(f"Optimizer {self.cfg.training.optimizer} not implemented")
        LOGGER.info(f"Using optimizer {self.cfg.training.optimizer} with lr={self.cfg.training.lr}")

    def _init_dataloader(self):
        n_data = self.particles.shape[0]
        self.split = int(n_data * self.cfg.training.split)

        train_dataset = AmplitudeDataset(self.particles_prepd[:self.split], self.amplitudes_prepd[:self.split],
                                         dtype=self.dtype)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.cfg.training.batchsize, shuffle=True)

        test_dataset = AmplitudeDataset(self.particles_prepd[self.split:], self.amplitudes_prepd[self.split:],
                                        dtype=self.dtype)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                        batch_size=self.cfg.evaluation.batchsize, shuffle=False)
        LOGGER.info(f"Constructed dataloaders with split={self.cfg.training.split}, "\
                    f"batch_size={self.cfg.training.batchsize}")

    def _init_scheduler(self):
        if self.cfg.training.scheduler is None:
            pass # constant learning rate
        elif self.cfg.training.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 self.cfg.training.lr * 10,
                                                                 epochs=self.cfg.training.nepochs,
                                                                 steps_per_epoch=len(self.train_loader))
        else:
            raise ValueError(f"Learning rate scheduler {self.cfg.training.scheduler} not implemented")

        LOGGER.info(f"Using learning rate scheduler {self.cfg.training.scheduler}")

    def train(self):
        # performance metrics
        self.metrics = {"loss": [], "lr": []}
        self.loss = torch.nn.MSELoss()
        
        # main train loop
        LOGGER.info(f"Starting to train for {self.cfg.training.nepochs} epochs")
        t00 = time.time()
        for epoch in range(self.cfg.training.nepochs):
            t0 = time.time()
            self.model.train()
            for data in self.train_loader:
                self._step(data)

            dt = time.time() - t0
            if epoch==0:
                LOGGER.info(f"Finished first epoch after {dt:.2f}s, "\
                            f"training time estimate: {dt*self.cfg.training.nepochs/60:.2f}min "\
                            f"= {dt*self.cfg.training.nepochs/60**2:.2f}h")
            else:
                LOGGER.debug(f"Finished epoch {epoch} after {dt:.2f}s")

        dt = time.time() - t00
        LOGGER.info(f"Finished training after {dt/60:.2f}min = {dt/60**2:.2f}h")

    def _step(self, data):
        x, y = data
        x, y = x.to(self.device), y.to(self.device)

        y_pred = self.model(x, type_token=self.type_token)
        loss = self.loss(y_pred, y)
        assert torch.isfinite(loss).all()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.metrics["loss"].append(loss.item())
        self.metrics["lr"].append(self.optimizer.param_groups[0]["lr"])

    def _save_model(self):
        filename = f"model_{self.cfg.exp_idx}.pt"
        model_path = os.path.join(self.cfg.exp_dir, "models", filename)
        LOGGER.info(f"Saving model at {model_path}")
        torch.save(self.model.state_dict(), model_path)
