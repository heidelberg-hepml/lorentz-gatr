import numpy as np
import torch

import os, sys, time
from omegaconf import OmegaConf, open_dict
from matplotlib.backends.backend_pdf import PdfPages

from experiments.base_experiment import BaseExperiment
from experiments.base_plots import plot_loss
from experiments.toptagging.wrappers import TopTaggingTransformerWrapper, TopTaggingGATrWrapper
from experiments.toptagging.dataset import TopTaggingDataset
#from experiments.toptagging.plots import plot_roc
from experiments.misc import get_device, flatten_dict
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP", "CLSTr": "CLSTr", "GAP": "GAP"}

class TopTaggingExperiment(BaseExperiment):

    def __init__(self, cfg):
        super().__init__(cfg)

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
        self.train_metrics = self._evaluate_single(self.train_loader, "train")
        self.test_metrics = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        return None

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        if self.cfg.plotting.loss and self.cfg.train:
            file = f"{plot_path}/loss.pdf"
            plot_loss(file, [self.train_loss, self.val_loss], self.train_lr,
                      labels=["train loss", "val loss"], logy=False)

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
