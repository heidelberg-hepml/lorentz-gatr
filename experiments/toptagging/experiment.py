import numpy as np
import torch
from torch_geometric.loader import DataLoader

import os, time
from omegaconf import OmegaConf, open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

import matplotlib.pyplot as plt

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP"}


class TopTaggingExperiment(BaseExperiment):
    def init_physics(self):
        if (
            self.cfg.model._target_
            == "experiments.toptagging.wrappers.TopTaggingTransformerWrapper"
        ):
            assert (
                not self.cfg.data.add_pairs
            ), "data.add_pairs are not implemented for the default Transformer"
            # assert not self.cfg.model.beam_reference, "model.beam_reference are not implemented for the default Transformer"

        if not self.cfg.training.force_xformers:
            LOGGER.warning(
                f"Using training.force_xformers=False, this will slow down the network by a factor of 5-10."
            )

        with open_dict(self.cfg):
            # extra mv channels for GATr
            if (
                self.cfg.model._target_
                == "experiments.toptagging.wrappers.TopTaggingGATrWrapper"
            ):
                if self.cfg.model.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += 1
                if self.cfg.data.add_pairs:
                    self.cfg.model.net.in_mv_channels += 2

    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz"
        )
        LOGGER.info(f"Loading top-tagging dataset from {data_path}")
        t0 = time.time()
        self.data_train = TopTaggingDataset(
            data_path,
            "train",
            data_scale=None,
            add_pairs=self.cfg.data.add_pairs,
            dtype=self.dtype,
        )
        self.data_test = TopTaggingDataset(
            data_path,
            "test",
            data_scale=self.data_train.data_scale,
            add_pairs=self.cfg.data.add_pairs,
            dtype=self.dtype,
        )
        self.data_val = TopTaggingDataset(
            data_path,
            "val",
            data_scale=self.data_train.data_scale,
            add_pairs=self.cfg.data.add_pairs,
            dtype=self.dtype,
        )
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def evaluate(self):
        self.results = {}
        self.results["train"] = self._evaluate_single(
            self.train_loader, "train", mode="eval"
        )
        self.results["val"] = self._evaluate_single(self.val_loader, "val", mode="eval")
        self.results["test"] = self._evaluate_single(
            self.test_loader, "test", mode="eval"
        )

    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]
        # re-initialize dataloader to make sure it is using the evaluation batchsize (makes a difference for trainloader)
        loader = DataLoader(
            dataset=loader.dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        if mode == "eval":
            LOGGER.info(
                f"### Starting to evaluate model on {title} dataset with "
                f"{len(loader.dataset.data_list)} elements, batchsize {loader.batch_size} ###"
            )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                y_pred = self.model(batch)
                y_pred = torch.nn.functional.sigmoid(y_pred)
                labels_true.append(batch.label.cpu().float())
                labels_predict.append(y_pred.cpu().float())
                break
        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # bce loss
        metrics["bce"] = torch.nn.functional.binary_cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = labels_true.numpy(), labels_predict.numpy()

        # accuracy
        metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict))
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # roc (fpr = epsB, tpr = epsS)
        fpr, tpr, th = roc_curve(labels_true, labels_predict)
        if mode == "eval":
            metrics["fpr"], metrics["tpr"] = fpr, tpr
        metrics["auc"] = roc_auc_score(labels_true, labels_predict)
        if mode == "eval":
            LOGGER.info(f"AUC score on {title} dataset: {metrics['auc']:.4f}")

        # 1/epsB at fixed epsS
        def get_rej(epsS):
            idx = np.argmin(np.abs(tpr - epsS))
            return 1 / fpr[idx]

        metrics["rej03"] = get_rej(0.3)
        metrics["rej05"] = get_rej(0.5)
        metrics["rej08"] = get_rej(0.8)
        if mode == "eval":
            LOGGER.info(
                f"Rejection rate {title} dataset: {metrics['rej03']:.0f} (epsS=0.3), "
                f"{metrics['rej05']:.0f} (epsS=0.5), {metrics['rej08']:.0f}"
            )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)
        return metrics

    def plot(self):
        plot_path = os.path.join(self.cfg.run_dir, f"plots_{self.cfg.run_idx}")
        os.makedirs(plot_path)
        model_title = MODEL_TITLE_DICT[type(self.model.net).__name__]
        title = model_title
        LOGGER.info(f"Creating plots in {plot_path}")

        if self.cfg.evaluate:
            file = f"{plot_path}/roc.txt"
            roc = np.stack(
                (self.results["test"]["fpr"], self.results["test"]["tpr"]), axis=-1
            )
            np.savetxt(file, roc)

        plot_dict = {}
        if self.cfg.evaluate:
            plot_dict = {"results_test": self.results["test"]}
        if self.cfg.train:
            plot_dict["train_loss"] = self.train_loss
            plot_dict["val_loss"] = self.val_loss
            plot_dict["train_lr"] = self.train_lr
            plot_dict["train_metrics"] = self.train_metrics
            plot_dict["val_metrics"] = self.val_metrics
        plot_mixer(self.cfg, plot_path, title, plot_dict)

    def _init_loss(self):
        self.loss = torch.nn.BCEWithLogitsLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        metrics = self._evaluate_single(self.val_loader, "val", mode="val")
        self.val_loss.append(metrics["bce"])
        return metrics["bce"]

    def _batch_loss(self, batch):
        batch = batch.to(self.device)
        y_pred = self.model(batch)
        loss = self.loss(y_pred, batch.label.to(self.dtype))
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _init_metrics(self):
        return {}
