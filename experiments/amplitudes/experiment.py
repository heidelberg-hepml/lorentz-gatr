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

from experiments.amplitudes.wrappers import AmplitudeMLPWrapper, AmplitudeTransformerWrapper, AmplitudeGATrWrapper
from experiments.baselines import MLP, BaselineTransformer
from experiments.amplitudes.dataset import AmplitudeDataset
from experiments.amplitudes.preprocessing import preprocess_particles, preprocess_amplitude, undo_preprocess_amplitude
from experiments.amplitudes.plots import plot_histograms, plot_loss, plot_single_histogram
from experiments.misc import get_device
import experiments.logger
from experiments.logger import LOGGER, MEMORY_HANDLER

from experiments.baselines import MLP, BaselineTransformer
from gatr.layers import MLPConfig, SelfAttentionConfig
cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)

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
        t0 = time.time()
        
        # initialize environment
        self._init_experiment()
        self._init_directory()
        self._init_logger()
        self._init_backend()

        self.init_model()
        self.init_data()
        self._init_dataloader()

        if self.cfg.train:  
            self.train()

        if self.cfg.evaluate:
            self.evaluate()

        if self.cfg.plot:
            self.plot()

        dt = time.time() - t0
        LOGGER.info(f"Finished experiment after {dt/60:.2f}min = {dt/60**2:.2f}h")

    def init_model(self):
        # initialize model
        self.model = instantiate(self.cfg.model) # hydra magic
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        LOGGER.info(f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters")

        # load existing model if specified
        if self.warm_start:
            warm_start_idx = 0 if self.cfg.warm_start_idx is None else self.cfg.warm_start_idx
            model_path = os.path.join(self.cfg.warm_start_path, "models", f"model_{warm_start_idx}.pt")
            try:
                state_dict = torch.load(model_path, map_location="cpu")
            except FileNotFoundError:
                raise ValueError(f"Cannot load model from {model_path}")
            LOGGER.info(f"Loaded model from {model_path}")
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def init_data(self):
        type_token_dict = {"aag": [0,1,2,2,3], "aagg": [0,1,2,2,3,3],
                           "zjj": [0,1,2,3,3], "zjjj": [0,1,2,3,3,3],
                           "zjjjj": [0,1,2,3,3,3,3]}
        self.type_token = type_token_dict[self.cfg.data.dataset]
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
        if type(self.model.net) in [MLP, BaselineTransformer]:
            self.particles_prepd, self.particles_mean, self.particles_std = preprocess_particles(self.particles)
        else:
            self.particles_prepd = self.particles / self.particles.std()

    def train(self):
        self._init_optimizer()
        self._init_scheduler()

        self._train()
        self._save_model()

    def evaluate(self):
        # compute predictions
        self.amplitudes_truth_prepd = self.test_loader.dataset.amplitudes.numpy()
        self.amplitudes_pred_prepd = np.zeros((0, 1))
        LOGGER.info(f"Starting to evaluate model on test dataset with {self.amplitudes_truth_prepd.shape[0]} elements")
        with torch.no_grad():
            for x, y in self.test_loader:
                y_pred = self.model(x, type_token=self.type_token)
                self.amplitudes_pred_prepd = np.concatenate((self.amplitudes_pred_prepd,
                                                             y_pred.cpu().numpy()), axis=0)
        assert self.amplitudes_truth_prepd.shape == self.amplitudes_pred_prepd.shape

        # compute metrics over preprocessed amplitudes 
        self.mse_prepd = np.mean( (self.amplitudes_truth_prepd - self.amplitudes_pred_prepd) **2)
        self.mae_prepd = np.linalg.norm(self.amplitudes_truth_prepd - self.amplitudes_pred_prepd, axis=1).mean()
        LOGGER.info(f"Metrics on test dataset (preprocessed): \tMSE={self.mse_prepd:.4e}, MAE={self.mae_prepd:.4e}")

        # undo preprocessing
        self.amplitudes_truth = undo_preprocess_amplitude(self.amplitudes_truth_prepd,
                                                          self.amplitudes_mean, self.amplitudes_std)
        self.amplitudes_pred = undo_preprocess_amplitude(self.amplitudes_pred_prepd,
                                                          self.amplitudes_mean, self.amplitudes_std)

        # compute metrics
        self.mse = np.mean( (self.amplitudes_truth - self.amplitudes_pred) **2)
        self.mae = np.linalg.norm(self.amplitudes_truth - self.amplitudes_pred, axis=1).mean()
        LOGGER.info(f"Metrics on test dataset: \t\t\tMSE={self.mse:.4e}, MAE={self.mae:.4e}")

    def plot(self):
        plot_path = os.path.join(self.exp_dir, f"plots_{self.cfg.exp_idx}")
        os.makedirs(plot_path)
        dataset_title = {"aag": r"$\gamma\gamma g$", "aagg": r"$\gamma\gamma gg$",
                              "zjj": "$Zjj$", "zjjj": "$Zjjj$", "zjjjj": "$Zjjjj$"}[self.cfg.data.dataset]
        model_title = {"GATr": "GATr", "BaselineTransformer": "Tr", "MLP": "MLP"}[type(self.model.net).__name__]
        title = f"{model_title}: {dataset_title}"
        
        if self.cfg.plotting.loss:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.metrics["loss"]], self.metrics["lr"], labels=["loss"])

        if self.cfg.plotting.histograms:
            out = f"{plot_path}/histograms.pdf"
            with PdfPages(out) as file:
                labels = ["Test", "Train", "Prediction"]

                data = [np.log(self.amplitudes_truth), np.log(self.amplitudes[:self.split]),
                        np.log(self.amplitudes_pred)]
                plot_histograms(file, data, labels, title=title,
                           xlabel=r"$\log A$", logx=False)

        if self.cfg.plotting.delta:
            out = f"{plot_path}/delta.pdf"
            with PdfPages(out) as file:
                data = (self.amplitudes_truth - self.amplitudes_pred) / self.amplitudes_truth
                plot_single_histogram(file, data, title=title,
                           xlabel=r"$\Delta = \frac{A_\mathrm{truth} - A_\mathrm{pred}}{A_\mathrm{truth}}$",
                           logx=False, xrange=(-.3, .3), bins=50)

    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_path is None else True

        if not self.warm_start:
            rnd_number = np.random.randint(low=0, high=9999)
            exp_name = f"{self.cfg.exp_name}_{rnd_number:04}"
            self.exp_dir = os.path.join(self.cfg.base_dir, "runs", exp_name)
            LOGGER.info(f"Creating new experiment {self.exp_dir}")
            
            self.exp_idx = 0
            
        else:
            config_path = os.path.join(self.cfg.warm_start_path, "config.yaml")
            try:
                warm_start_cfg = OmegaConf.load(config_path) # overwrite self.cfg
            except FileNotFoundError:
                raise ValueError(f"Cannot load config from {config_path}")
            self.cfg = OmegaConf.merge(warm_start_cfg, self.cfg) # warm_start_cfg overrides others
            self.exp_idx = self.cfg.exp_idx + 1
            LOGGER.info(f"Warm-starting from existing experiment {self.cfg.exp_dir} for {self.exp_idx}th time")

        OmegaConf.set_struct(self.cfg, True)
        with open_dict(self.cfg):
            self.cfg.exp_idx = self.exp_idx
            self.cfg.warm_start = self.warm_start
            if not self.warm_start:
                self.cfg.exp_dir = self.exp_dir

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
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
            
        self._save_config("config.yaml")
        self._save_config(f"config_{self.cfg.exp_idx}.yaml")

    def _init_logger(self):
        if experiments.logger.logging_initialized:
            LOGGER.info("Logger already initialized")
            return
        
        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        formatter = logging.Formatter("[%(asctime)-19.19s %(levelname)-1.1s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        
        # init file_handler
        file_handler = logging.FileHandler(Path(self.cfg.exp_dir) / f"out_{self.cfg.exp_idx}.log")
        file_handler.setFormatter(formatter)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

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
        
    def _save_config(self, filename="config.yaml"):
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

    def _train(self):
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
        loss = self.loss(y, y_pred)
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
