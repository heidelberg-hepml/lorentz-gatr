import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score

#from experiments.base_experiment import BaseExperiment
from experiments.base_plots import plot_loss
from experiments.toptagging.wrappers import TopTaggingTransformerWrapper, TopTaggingGATrWrapper
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.plots import plot_roc
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}

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

class TopTaggingExperiment:

    # generic code

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
            model_path = os.path.join(self.cfg.run_dir, "models", f"model_{self.cfg.warm_start_idx}.pt")
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
        smallest_val_loss = 1e10
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
                patience = 0
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

    def _save_model(self):
        if not self.cfg.save:
            return
        
        filename = f"model_{self.cfg.run_idx}.pt"
        model_path = os.path.join(self.cfg.run_dir, "models", filename)
        LOGGER.debug(f"Saving model at {model_path}")
        torch.save(self.model.state_dict(), model_path)

    # experiment-specific code

    def init_physics(self):
        pass

    def init_data(self):
        # load
        LOGGER.info(f"Loading top-tagging dataset from {self.cfg.data.data_path}")
        data = np.load(self.cfg.data.data_path)
        kinematics, labels = data["kinematics"], data["labels"]

        # preprocessing (= change units)
        self.kinematics_std = kinematics.std()
        kinematics = kinematics / self.kinematics_std


        # extract train, test, val (only save it once!)
        train_idx, test_idx, val_idx = (labels[:,0] == 0), (labels[:,0] == 1), (labels[:,0] == 2)
        self.data_train = TopTaggingDataset(kinematics[train_idx,...], labels[train_idx,1,None], self.dtype)
        self.data_test = TopTaggingDataset(kinematics[test_idx,...], labels[test_idx,1,None], self.dtype)
        self.data_val = TopTaggingDataset(kinematics[val_idx,...], labels[val_idx,1,None], self.dtype)

    def evaluate(self):
        #self.train_metrics = self._evaluate_single(self.train_loader, "train")
        self.fpr, self.tpr, self.auc = self.test_metrics = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        LOGGER.info(f"### Starting to evaluate model on {title} dataset with "\
                    f"{loader.dataset.kinematics.shape[0]} elements ###")

        # predictions
        labels_true, labels_predict = np.zeros((0, 1)), np.zeros((0, 1))
        self.model.eval()
        with torch.no_grad():
            for x, y, mask in loader:
                x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
                y_pred = self.model(x, attention_mask=mask)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                labels_true = np.concatenate((labels_true,
                                              y.cpu().float().numpy()), axis=0)
                labels_predict = np.concatenate((labels_predict,
                                              y_pred.cpu().float().numpy()), axis=0)
        assert labels_true.shape == labels_predict.shape

        # accuracy
        labels_predict_rounded = np.round(labels_predict)
        accuracy = (labels_predict_rounded == labels_true).sum() / labels_true.shape[0]
        LOGGER.info(f"Accuracy on {title} dataset: {accuracy:.4f}")
        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.accuracy", accuracy)

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true[:,0], labels_predict[:,0])
        auc = roc_auc_score(labels_true[:,0], labels_predict[:,0])
        LOGGER.info(f"AUC score on {title} dataset: {auc:.4f}")
        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.auc", auc)

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1/fpr[idx]
        rej03 = get_rej(0.3)
        rej05 = get_rej(0.5)
        LOGGER.info(f"Rejection rate {title} dataset: {rej03:.0f} (epsS=0.3), {rej05:.0f} (epsS=0.5)")
        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.rej03", rej03)
            log_mlflow(f"eval.{title}.rej05", rej05)

        return fpr, tpr, auc

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        file = f"{plot_path}/roc.txt"
        roc = np.stack((self.fpr, self.tpr), axis=-1)
        np.savetxt(file, roc)

        if self.cfg.plotting.loss and self.cfg.train:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.train_loss, self.val_loss], self.train_lr,
                      labels=["train loss", "val loss"], logy=False)

        if self.cfg.plotting.roc:
            file = f"{plot_path}/roc.pdf"
            with PdfPages(file) as out:
                plot_roc(out, self.fpr, self.tpr, self.auc)

    def _init_dataloader(self):
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data_train,
                                                        batch_size=self.cfg.training.batchsize, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.data_test,
                                                        batch_size=self.cfg.training.batchsize, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=self.data_val,
                                                        batch_size=self.cfg.training.batchsize, shuffle=True)

        LOGGER.debug(f"Constructed dataloaders with batch_size={self.cfg.training.batchsize}, "\
                     f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, ")

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    def _batch_loss(self, data):
        x, y, mask = data
        x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
        y_pred = self.model(x, attention_mask=mask)
        loss = self.loss(y_pred, y)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _init_metrics(self):
        return {}
