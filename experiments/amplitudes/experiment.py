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

# generic imports
import zipfile
import logging
from pathlib import Path
from omegaconf import OmegaConf, open_dict, errors
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import mlflow

from experiments.misc import get_device, flatten_dict
import experiments.logger
from experiments.logger import LOGGER, MEMORY_HANDLER, FORMATTER

from gatr.layers import MLPConfig, SelfAttentionConfig
cs = ConfigStore.instance()
cs.store(name="base_attention", node=SelfAttentionConfig)
cs.store(name="base_mlp", node=MLPConfig)

class AmplitudeExperiment:

    # generic methods (will be factored out; currently have technical issues with inheritance)

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        # pass all exceptions to the logger
        try:
            self.run_mlflow()
        except errors.ConfigAttributeError:
            LOGGER.exception("Tried to access key that is not specified in the config files")
        except:
            LOGGER.exception("Exiting with error")

        # print buffered logger messages if failed
        if not experiments.logger.LOGGING_INITIALIZED:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            MEMORY_HANDLER.setTarget(stream_handler)
            MEMORY_HANDLER.close()

    def run_mlflow(self):
        experiment_id, run_name = self._init()
        LOGGER.info(f"### Starting experiment {self.cfg.exp_name}/{run_name} (id={experiment_id}) ###")
        if self.cfg.use_mlflow:
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):            
                self.full_run()
        else:
            # dont use mlflow
            self.full_run()
                
    def full_run(self):
        # implement all ml boilerplate as private methods (_name)
        t0 = time.time()

        # save config
        LOGGER.debug(OmegaConf.to_yaml(self.cfg))
        self._save_config("config.yaml", to_mlflow=True)
        self._save_config(f"config_{self.cfg.run_idx}.yaml")

        self.init_physics()
        self.init_model()
        self.init_data()
        self._init_dataloader()

        if self.cfg.train:
            self._init_optimizer()
            self._init_scheduler()
            self._init_loss()
            self.train()
            self._save_model()

        if self.cfg.evaluate:
            self.evaluate()

        if self.cfg.plot and self.cfg.save:
            self.plot()

        dt = time.time() - t0
        LOGGER.info(f"Finished experiment after {dt/60:.2f}min = {dt/60**2:.2f}h")

    def init_model(self):
        # initialize model
        self.model = instantiate(self.cfg.model) # hydra magic
        num_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.cfg.use_mlflow:
            log_mlflow("num_parameters", float(num_parameters), step=0)
        LOGGER.info(f"Instantiated model {type(self.model.net).__name__} with {num_parameters} learnable parameters")

        # load existing model if specified
        if self.warm_start:
            model_path = os.path.join(self.cfg.run_dir, "models", f"model_run{self.cfg.warm_start_idx}.pt")
            try:
                state_dict = torch.load(model_path, map_location="cpu")
            except FileNotFoundError:
                raise ValueError(f"Cannot load model from {model_path}")
            LOGGER.info(f"Loading model from {model_path}")
            self.model.load_state_dict(state_dict)

        self.model.to(self.device, dtype=self.dtype)

    def _init(self):
        run_name = self._init_experiment()
        self._init_directory()

        if self.cfg.use_mlflow:
            experiment_id = self._init_mlflow()
        else:
            experiment_id = None
        
        # initialize environment
        self._init_logger()
        self._init_backend()

        return experiment_id, run_name
                    
    def _init_experiment(self):
        self.warm_start = False if self.cfg.warm_start_idx is None else True

        if not self.warm_start:
            if self.cfg.run_name is None:
                modelname = self.cfg.model.net._target_.rsplit(".", 1)[-1]
                rnd_number = np.random.randint(low=0, high=9999)
                run_name = f"{modelname}_{rnd_number:04}"
            else:
                run_name = self.cfg.run_name

            run_dir = os.path.join(self.cfg.base_dir, "runs", self.cfg.exp_name, run_name)
            run_idx = 0
            LOGGER.info(f"Creating new experiment {self.cfg.exp_name}/{run_name}")
            
        else:
            run_name = self.cfg.run_name
            run_idx = self.cfg.run_idx + 1
            LOGGER.info(f"Warm-starting from existing experiment {self.cfg.exp_name}/{run_name} for run {run_idx}")

        with open_dict(self.cfg):
            self.cfg.run_idx = run_idx
            if not self.warm_start:
                self.cfg.warm_start_idx = 0
                self.cfg.run_name = run_name
                self.cfg.run_dir = run_dir

            # only use mlflow if save=True
            self.cfg.use_mlflow = False if self.cfg.save==False else self.cfg.use_mlflow

        # set seed
        if self.cfg.seed is not None:
            LOGGER.info(f"Using seed {self.cfg.seed}")
            torch.random.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        return run_name

    def _init_mlflow(self):
        # mlflow tracking location
        Path(self.cfg.mlflow.db).parent.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{Path(self.cfg.mlflow.db).resolve()}")

        Path(self.cfg.mlflow.artifacts).mkdir(exist_ok=True)
        try:
            # artifacts not supported
            # mlflow call triggers alembic.runtime.migration logger to shout -> shut it down (happy for suggestions on how to do this nicer)
            logging.disable(logging.WARNING)
            experiment_id = mlflow.create_experiment(self.cfg.exp_name,
                        artifact_location=f"file:{Path(self.cfg.mlflow.artifacts).resolve()}")
            logging.disable(logging.DEBUG)
            LOGGER.info(f"Created mlflow experiment {self.cfg.exp_name} with id {experiment_id}")
        except mlflow.exceptions.MlflowException:
            LOGGER.info(f"Using existing mlflow experiment {self.cfg.exp_name}")
            logging.disable(logging.DEBUG)
        

        experiment = mlflow.set_experiment(self.cfg.exp_name)
        experiment_id = experiment.experiment_id

        LOGGER.info(f"Set experiment {self.cfg.exp_name} with id {experiment_id}")
        return experiment_id

    def _init_directory(self):
        if not self.cfg.save:
            LOGGER.info(f"Running with save=False, i.e. no outputs will be saved")
            return
        
        # create experiment directory
        run_dir = Path(self.cfg.run_dir).resolve()
        if run_dir.exists() and not self.warm_start:
            raise ValueError(f"Experiment in directory {self.cfg.run_dir} alredy exists. Aborting.")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

        # save source
        if self.cfg.save_source:
            zip_name = os.path.join(self.cfg.run_dir, "source.zip")
            LOGGER.debug(f"Saving source to {zip_name}")
            zipf = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
            path_gatr = os.path.join(self.cfg.base_dir, "gatr")
            path_experiment = os.path.join(self.cfg.base_dir, "experiments")
            for path in [path_gatr, path_experiment]:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, path))
            zipf.close()

    def _init_logger(self):
        # silence other loggers
        # (every app has a logger, eg hydra, torch, mlflow, matplotlib, fontTools...)
        for name, other_logger in logging.root.manager.loggerDict.items():
            if not "lorentz-gatr" in name:
                other_logger.level = logging.WARNING

        if experiments.logger.LOGGING_INITIALIZED:
            LOGGER.info("Logger already initialized")
            return
        
        LOGGER.setLevel(logging.DEBUG if self.cfg.debug else logging.INFO)
        
        # init file_handler
        if self.cfg.save:
            file_handler = logging.FileHandler(Path(self.cfg.run_dir) / f"out_{self.cfg.run_idx}.log")
            file_handler.setFormatter(FORMATTER)
            file_handler.setLevel(logging.DEBUG)
            LOGGER.addHandler(file_handler)

        # init stream_handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(LOGGER.level)
        stream_handler.setFormatter(FORMATTER)
        LOGGER.addHandler(stream_handler)

        # flush memory to stream_handler
        # this allows to catch logs that were created before the logger was initialized
        MEMORY_HANDLER.setTarget(stream_handler) # can only flush to one handler, choose stream_handler
        MEMORY_HANDLER.close()
        LOGGER.removeHandler(MEMORY_HANDLER)

        # add new handlers to logger
        LOGGER.propagate = False # avoid duplicate log outputs

        experiments.logger.LOGGING_INITIALIZED = True
        LOGGER.debug("Logger initialized")

    def _init_backend(self):
        self.device = get_device()
        LOGGER.info(f"Using device {self.device}")
        
        if self.cfg.training.float16 and self.device=="cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            LOGGER.debug("Using dtype bfloat16")
        elif self.cfg.training.float16:
            self.dtype = torch.float16
            LOGGER.debug("Using dtype float16 (bfloat16 is not supported by environment)")
        else:
            self.dtype = torch.float32
            LOGGER.debug("Using dtype float32")

        torch.backends.cuda.enable_flash_sdp(self.cfg.training.enable_flash_sdp)
        torch.backends.cuda.enable_math_sdp(self.cfg.training.enable_math_sdp)
        torch.backends.cuda.enable_mem_efficient_sdp(self.cfg.training.enable_mem_efficient_sdp)
        if self.cfg.training.force_xformers:
            LOGGER.debug("Forcing use of xformers' attention implementation")
            gatr.primitives.attention.FORCE_XFORMERS = True

    def _init_optimizer(self):
        if self.cfg.training.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.cfg.training.lr,
                                              betas=self.cfg.training.betas,
                                              eps=self.cfg.training.eps)
        else:
            raise ValueError(f"Optimizer {self.cfg.training.optimizer} not implemented")
        LOGGER.debug(f"Using optimizer {self.cfg.training.optimizer} with lr={self.cfg.training.lr}")

    def _init_scheduler(self):
        if self.cfg.training.scheduler is None:
            self.scheduler = None # constant lr
        elif self.cfg.training.scheduler == "OneCycleLR":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                                 self.cfg.training.lr * 10,
                                                                 epochs=self.cfg.training.nepochs,
                                                                 steps_per_epoch=len(self.train_loader))
        else:
            raise ValueError(f"Learning rate scheduler {self.cfg.training.scheduler} not implemented")

        LOGGER.debug(f"Using learning rate scheduler {self.cfg.training.scheduler}")

    def train(self):
        # performance metrics
        self.train_lr, self.train_loss, self.val_loss = [], [], []
        self.train_metrics = self._init_metrics()
        self.val_metrics = self._init_metrics()

        # early stopping
        smallest_val_loss, smallest_val_loss_epoch = 1e10, 0
        patience = 0
        
        # main train loop
        LOGGER.info(f"Starting to train for {self.cfg.training.nepochs} epochs "\
                    f"using early stopping with patience {self.cfg.training.es_patience}")
        self.training_start_time = time.time()
        for epoch in range(self.cfg.training.nepochs):
            t0 = time.time()

            # train
            self.model.train()
            for it, data in enumerate(self.train_loader):
                self._step(data, epoch * len(self.train_loader) + it)

            # validate
            val_loss = self._validate(epoch)
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                smallest_val_loss_epoch = epoch
                patience = 0

                # save best model (only in 2nd half of training to save time and disk space)
                if self.cfg.training.es_load_best_model and epoch > self.cfg.training.nepochs / 2:
                    self._save_model(f"model_run{self.cfg.run_idx}_ep{smallest_val_loss_epoch}.pt")
            else:
                patience += 1
                if patience > self.cfg.training.es_patience:
                    LOGGER.info(f"Early stopping in epoch {epoch}")
                    break

            # output
            dt = time.time() - t0
            if epoch==0:
                LOGGER.info(f"Finished first epoch after {dt:.2f}s, "\
                            f"training time estimate: {dt*self.cfg.training.nepochs/60:.2f}min "\
                            f"= {dt*self.cfg.training.nepochs/60**2:.2f}h")
            else:
                LOGGER.debug(f"Finished epoch {epoch} after {dt:.2f}s with val_loss={val_loss:.4f}")

        dt = time.time() - self.training_start_time
        LOGGER.info(f"Finished training after {dt/60:.2f}min = {dt/60**2:.2f}h")
        if self.cfg.use_mlflow:
            log_mlflow("es_epoch", epoch)
            log_mlflow("traintime", dt / 3600)

        # wrap up early stopping
        if self.cfg.training.es_load_best_model:
            model_path = os.path.join(self.cfg.run_dir, "models",
                                      f"model_run{self.cfg.run_idx}_ep{smallest_val_loss_epoch}.pt")
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                LOGGER.info(f"Loading model from {model_path}")
                self.model.load_state_dict(state_dict)
            except FileNotFoundError:
                LOGGER.warning(f"Cannot load best model (epoch {smallest_val_loss_epoch}) from {model_path}")

    def _step(self, data, step):
        loss, metrics = self._batch_loss(data)
        
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.training.clip_grad_norm,
                                                   error_if_nonfinite=True).cpu().item()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.train_loss.append(loss.item())
        self.train_lr.append(self.optimizer.param_groups[0]["lr"])
        for key, value in metrics.items():
            self.train_metrics[key].append(value)

        # log to mlflow
        if self.cfg.use_mlflow and self.cfg.training.log_every_n_steps!=0 \
           and step%self.cfg.training.log_every_n_steps==0:
            log_dict = {"loss": loss.item(), "lr": self.train_lr[-1],
                        "time_per_step": (time.time() - self.training_start_time) / (step+1),
                        "grad_norm": grad_norm}
            for key, values in log_dict.items():
                log_mlflow(f"train.{key}", values, step=step)

            for key, values in metrics.items():
                log_mlflow(f"train.{key}", values, step=step)
                

    def _validate(self, epoch):
        losses = []
        metrics = self._init_metrics()
        
        self.model.eval()
        with torch.no_grad():
            for data in self.val_loader:
                loss, metric = self._batch_loss(data)
                losses.append(loss.cpu().item())
                for key, value in metric.items():
                    metrics[key].append(value)
        val_loss = np.mean(losses)
        self.val_loss.append(val_loss)
        for key, values in metrics.items():
            self.val_metrics[key].append(np.mean(values))
        if self.cfg.use_mlflow:
            log_mlflow("val.loss", val_loss, step=epoch)
            for key, values in self.val_metrics.items():
                log_mlflow(f"val.{key}", values[-1], step=epoch)
        return val_loss

    def _save_config(self, filename="amplitudes.yaml", to_mlflow=False):
        # Save config
        if not self.cfg.save:
            return
        
        config_filename = Path(self.cfg.run_dir) / filename
        LOGGER.debug(f"Saving config at {config_filename}")
        with open(config_filename, "w", encoding="utf-8") as file:
            file.write(OmegaConf.to_yaml(self.cfg))

        if to_mlflow and self.cfg.use_mlflow:
            for key, value in flatten_dict(self.cfg).items():
                log_mlflow(key, value, kind="param")

    def _save_model(self, filename=None):
        if not self.cfg.save:
            return
        
        if filename is None:
            filename = f"model_run{self.cfg.run_idx}.pt"
        model_path = os.path.join(self.cfg.run_dir, "models", filename)
        LOGGER.debug(f"Saving model at {model_path}")
        torch.save(self.model.state_dict(), model_path)

    ### experiment-specific methods

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
        assert sum(self.cfg.training.train_test_val) <= 1

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

        LOGGER.info(f"Constructed dataloaders with train_test_val={self.cfg.training.train_test_val}, "\
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
            rmse_prepd = np.mean( ( (amp_pred_prepd - amp_truth_prepd) / amp_truth_prepd) **2)

            # undo preprocessing
            amp_truth = undo_preprocess_amplitude(amp_truth_prepd,
                                                     self.prepd_mean[idataset], self.prepd_std[idataset])
            amp_pred = undo_preprocess_amplitude(amp_pred_prepd,
                                                  self.prepd_mean[idataset], self.prepd_std[idataset])

            # compute metrics over actual amplitudes
            mse = np.mean( (amp_truth - amp_pred)**2 )
            delta = (amp_truth - amp_pred) / amp_truth
            rmse = np.mean( delta**2)

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
                        f"eval.{title}.{dataset}.rmse": rmse_prepd,
                        f"eval.{title}.{dataset}.mse_raw": mse,
                        f"eval.{title}.{dataset}.rmse_raw": rmse}
                if self.cfg.heteroscedastic:
                    log_dict[f"eval.{title}.{dataset}.pull_mean"] = np.mean(pull_prepd)
                    log_dict[f"eval.{title}.{dataset}.pull_std"] = np.std(pull_prepd)
                    log_dict[f"eval.{title}.{dataset}.pull_mean_raw"] = np.mean(pull)
                    log_dict[f"eval.{title}.{dataset}.pull_std_raw"] = np.std(pull)
                for key, value in log_dict.items():
                    log_mlflow(key, value)

            amp = {"raw": {"truth": amp_truth, "prediction": amp_pred, "mse": mse,
                       "rmse": rmse, "pull": pull},
               "preprocessed": {"truth": amp_truth_prepd, "prediction": amp_pred_prepd,
                                "mse": mse_prepd, "rmse": rmse_prepd, "pull": pull_prepd}}
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
