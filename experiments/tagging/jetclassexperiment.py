import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import open_dict

import os, time

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

from experiments.tagging.experiment import TaggingExperiment
from experiments.tagging.embedding import (
    dense_to_sparse_jet,
    embed_tagging_data_into_ga,
)

from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class JetClassTaggingExperiment(TaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            not self.cfg.model.mean_aggregation
        ), "Mean-aggregation not implemented for multi-class classification"
        with open_dict(self.cfg):
            if self.cfg.data.score_token:
                raise NotImplementedError
            else:
                self.cfg.data.num_global_tokens = 1
                self.cfg.model.net.out_mv_channels = 10

            if self.cfg.data.all_features:
                self.cfg.model.net.in_s_channels = 10
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/allfeatures.yaml"
                )
            else:
                self.cfg.model.net.in_s_channels = 0
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/fourmomenta.yaml"
                )

    def _init_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def init_data(self):
        LOGGER.info(f"Creating SimpleIterDataset")
        t0 = time.time()

        classes = [
            "HToBB",
            "HToCC",
            "HToGG",
            "HToWW2Q1L",
            "HToWW4Q",
            "TTBar",
            "TTBarLep",
            "WToQQ",
            "ZToQQ",
            "ZJetsToNuNu",
        ]
        frange = (0, 1)
        datasets = {"train": None, "test": None, "val": None}
        self.num_files = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_100M", "test": "test_20M", "val": "val_5M"}
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [f"{classname}:{path}/{classname}_*.root" for classname in classes]
            file_dict, files = to_filelist(flist)
            self.num_files[label] = len(files)

            LOGGER.info(
                f"Using {self.num_files[label]} files for {label}ing, range: {str(frange)}"
            )
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.data_config,
                for_training=True,
                extra_selection=self.cfg.jc_params.extra_selection,
                remake_weights=not self.cfg.jc_params.not_remake_weights,
                load_range_and_fraction=(frange, self.cfg.jc_params.data_fraction),
                file_fraction=self.cfg.jc_params.file_fraction,
                fetch_by_files=self.cfg.jc_params.fetch_by_files,
                fetch_step=self.cfg.jc_params.fetch_step,
                infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
                in_memory=self.cfg.jc_params.in_memory,
                name=label,
            )
        self.data_train = datasets["train"]
        self.data_test = datasets["test"]
        self.data_val = datasets["val"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            drop_last=True,
            pin_memory=True,
            num_workers=min(
                self.cfg.jc_params.num_workers,
                int(self.num_files["train"] * self.cfg.jc_params.file_fraction),
            ),
            persistent_workers=self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch is not None,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=True,
            pin_memory=True,
            num_workers=min(
                self.cfg.jc_params.num_workers,
                int(self.num_files["val"] * int(self.cfg.jc_params.file_fraction)),
            ),
            persistent_workers=self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch_val is not None,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=False,
            pin_memory=True,
            num_workers=min(self.cfg.jc_params.num_workers, self.num_files["test"]),
        )

    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]
        # re-initialize dataloader to make sure it is using the evaluation batchsize
        # (makes a difference for trainloader)
        loader = DataLoader(
            dataset=loader.dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        if mode == "eval":
            LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            for batch in loader:
                y_pred, label = self._get_ypred_and_label(batch)
                labels_true.append(label.cpu())
                labels_predict.append(y_pred.cpu().float())

        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )

        # ce loss
        metrics["bce"] = torch.nn.functional.cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = (
            labels_true.numpy(),
            torch.softmax(labels_predict, dim=1).numpy(),
        )

        # accuracy
        labels_predict_score = np.argmax(labels_predict, axis=1)
        metrics["accuracy"] = accuracy_score(
            labels_true.flatten(), np.round(labels_predict_score).flatten()
        )
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

        # auc and roc (fpr = epsB, tpr = epsS)
        metrics["auc_ovo"] = roc_auc_score(
            labels_true, labels_predict, multi_class="ovo", average="macro"
        )
        if mode == "eval":
            LOGGER.info(f"The AUC is {metrics['auc_ovo']}")
        fpr_list, tpr_list, auc_scores = [], [], []
        for i in range(self.cfg.jc_params.num_classes):
            fpr, tpr, _ = roc_curve(labels_true == i, labels_predict[:, i])
            auc_score = roc_auc_score(labels_true == i, labels_predict[:, i])
            auc_scores.append(auc_score)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            metrics["auc_class_{}".format(i)] = auc_score
            if mode == "eval":
                LOGGER.info(
                    f"AUC score for class {i} on {title} dataset: {auc_score:.4f}"
                )

        metrics["auc_total"] = np.mean(auc_scores)
        # 1/epsB at fixed epsS
        def get_rej(epsS, class_idx):
            idx = np.argmin(np.abs(tpr_list[class_idx] - epsS))
            return 1 / fpr_list[class_idx][idx]

        for i in range(self.cfg.jc_params.num_classes):
            metrics["rej05_{}".format(i)] = get_rej(0.5, i)
            metrics["rej099_{}".format(i)] = get_rej(0.99, i)
            metrics["rej0995_{}".format(i)] = get_rej(0.995, i)
            if mode == "eval":
                LOGGER.info(
                    f"Rejection rate for class {i} on {title} dataset: {metrics[f'rej05_{i}']:.0f} (epsS=0.5), "
                    f"{metrics[f'rej099_{i}']:.0f} (epsS=0.99), {metrics[f'rej0995_{i}']:.0f} (epsS=0.995)"
                )

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict", "fpr", "tpr"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        return metrics

    def _get_ypred_and_label(self, batch):
        fourmomenta = batch[0]["pf_vectors"].to(self.device)
        if self.cfg.data.all_features:
            scalars = batch[0]["pf_features"].to(self.device)
        else:
            scalars = torch.empty(
                fourmomenta.shape[0],
                0,
                fourmomenta.shape[2],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
        label = batch[1]["_label_"].to(self.device)
        fourmomenta, scalars, ptr = dense_to_sparse_jet(fourmomenta, scalars)
        embedding = embed_tagging_data_into_ga(fourmomenta, scalars, ptr, self.cfg.data)
        y_pred = self.model(embedding)
        return y_pred, label
