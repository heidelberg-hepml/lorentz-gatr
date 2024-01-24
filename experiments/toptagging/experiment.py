import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import OmegaConf, open_dict

from sklearn.metrics import roc_curve, roc_auc_score

from experiments.base_experiment import BaseExperiment
from experiments.toptagging.wrappers import TopTaggingTransformerWrapper, TopTaggingGATrWrapper
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr"}

class TopTaggingExperiment(BaseExperiment):

    def init_physics(self):
        assert self.cfg.training.force_xformers, "xformers attention required for toptagging"

    def init_data(self):
        LOGGER.info(f"Loading top-tagging dataset from {self.cfg.data.data_path}")
        t0 = time.time()
        self.data_train = TopTaggingDataset(self.cfg.data.data_path, "train",
                                            data_scale=None, dtype=self.dtype)
        self.data_test = TopTaggingDataset(self.cfg.data.data_path, "test",
                                           data_scale=self.data_train.data_scale, dtype=self.dtype)
        self.data_val = TopTaggingDataset(self.cfg.data.data_path, "val",
                                          data_scale=self.data_train.data_scale, dtype=self.dtype)
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.train_loader = DataLoader(dataset=self.data_train,
                                        batch_size=self.cfg.training.batchsize, shuffle=True)
        self.test_loader = DataLoader(dataset=self.data_test,
                                        batch_size=self.cfg.training.batchsize, shuffle=False)
        self.val_loader = DataLoader(dataset=self.data_val,
                                        batch_size=self.cfg.training.batchsize, shuffle=False)

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def evaluate(self):
        self.results_train = self._evaluate_single(self.train_loader, "train")
        self.results_val = self._evaluate_single(self.val_loader, "val")
        self.results_test = self._evaluate_single(self.test_loader, "test")

    def _evaluate_single(self, loader, title):
        LOGGER.info(f"### Starting to evaluate model on {title} dataset with "\
                    f"{len(loader.dataset.data_list)} elements, batchsize {loader.batch_size} ###")

        # predictions
        labels_true, labels_predict = np.zeros((0)), np.zeros((0))
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                y_pred = self.model(batch)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                labels_true = np.concatenate((labels_true,
                                              batch.label.cpu().float().numpy()), axis=0)
                labels_predict = np.concatenate((labels_predict,
                                              y_pred.cpu().float().numpy()), axis=0)
        assert labels_true.shape == labels_predict.shape

        # accuracy
        LOGGER.info(f"{labels_true.mean()} {labels_true.std()} {labels_predict.mean()} {labels_predict.std()}")
        labels_predict_rounded = np.round(labels_predict)
        accuracy = (labels_predict_rounded == labels_true).sum() / labels_true.shape[0]
        LOGGER.info(f"Accuracy on {title} dataset: {accuracy:.4f}")
        if self.cfg.use_mlflow:
            log_mlflow(f"eval.{title}.accuracy", accuracy)

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        auc = roc_auc_score(labels_true, labels_predict)
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

        results = {"fpr": fpr, "tpr": tpr, "auc": auc}
        return results

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        file = f"{plot_path}/roc.txt"
        roc = np.stack((self.results_test["fpr"], self.results_test["tpr"]), axis=-1)
        np.savetxt(file, roc)

        plot_dict = {"results_test": self.results_test}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    def _batch_loss(self, batch):
        batch = batch.to(self.device)
        y_pred = self.model(batch)
        loss = self.loss(y_pred, batch.label)
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _init_metrics(self):
        return {}
