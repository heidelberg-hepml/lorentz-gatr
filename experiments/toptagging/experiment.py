import numpy as np
import torch

import os, time
from omegaconf import OmegaConf, open_dict
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import roc_curve, roc_auc_score

from experiments.base_experiment import BaseExperiment
from experiments.base_plots import plot_loss
from experiments.toptagging.wrappers import TopTaggingTransformerWrapper, TopTaggingGATrWrapper
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.plots import plot_roc
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}

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
        log_mlflow(f"eval.{title}.accuracy", accuracy)

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true[:,0], labels_predict[:,0])
        auc = roc_auc_score(labels_true[:,0], labels_predict[:,0])
        LOGGER.info(f"AUC score on {title} dataset: {auc:.4f}")
        log_mlflow(f"eval.{title}.auc", auc)

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1/fpr[idx]
        rej03 = get_rej(0.3)
        rej05 = get_rej(0.5)
        LOGGER.info(f"Rejection rate {title} dataset: {rej03:.0f} (epsS=0.3), {rej05:.0f} (epsS=0.5)")
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
