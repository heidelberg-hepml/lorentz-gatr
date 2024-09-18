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
        assert not self.cfg.plotting.roc and not self.cfg.plotting.score
        assert self.cfg.jc_params.num_classes == 10
        with open_dict(self.cfg):
            if self.cfg.data.score_token:
                self.cfg.data.num_global_tokens = self.cfg.jc_params.num_classes
                self.cfg.model.net.out_mv_channels = 1
            else:
                self.cfg.data.num_global_tokens = 1
                self.cfg.model.net.out_mv_channels = self.cfg.jc_params.num_classes

            if self.cfg.data.features == "fourmomenta":
                self.cfg.model.net.in_s_channels = 0
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/fourmomenta.yaml"
                )
            elif self.cfg.data.features == "pid":
                self.cfg.model.net.in_s_channels = 6
                self.cfg.data.data_config = "experiments/tagging/miniweaver/pid.yaml"
            elif self.cfg.data.features == "default":
                self.cfg.model.net.in_s_channels = 10
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/default.yaml"
                )
            elif self.cfg.data.features == "kitchensink":
                self.cfg.model.net.in_s_channels = 17
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/kitchensink.yaml"
                )
            else:
                raise ValueError(
                    f"Input feature option {self.cfg.data.features} not implemented"
                )

            self.class_names = [
                "ZJetsToNuNu",
                "HToBB",
                "HToCC",
                "HToGG",
                "HToWW4Q",
                "HToWW2Q1L",
                "TTBar",
                "TTBarLep",
                "WToQQ",
                "ZToQQ",
            ]

    def _init_loss(self):
        self.loss = torch.nn.CrossEntropyLoss()

    def init_data(self):
        LOGGER.info(f"Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"train": None, "test": None, "val": None}

        for_training = {"train": True, "val": True, "test": False}
        folder = {"train": "train_100M", "test": "test_20M", "val": "val_5M"}
        files_range = {
            "train": self.cfg.data.train_files_range,
            "test": self.cfg.data.test_files_range,
            "val": self.cfg.data.val_files_range,
        }
        self.num_files = {
            label: frange[1] - frange[0] for label, frange in files_range.items()
        }
        for label in ["train", "test", "val"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{path}/{classname}_{str(i).zfill(3)}.root"
                for classname in self.class_names
                for i in range(*files_range[label])
            ]
            file_dict, files = to_filelist(flist)

            LOGGER.info(
                f"Using {self.num_files[label]} files for {label}ing from {path}"
            )
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.data_config,
                for_training=True,
                extra_selection=self.cfg.jc_params.extra_selection,
                remake_weights=not self.cfg.jc_params.not_remake_weights,
                load_range_and_fraction=((0, 1), 1),
                file_fraction=1,
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
        self.loader_kwargs = {
            "pin_memory": True,
            "num_workers": self.cfg.jc_params.num_workers,
            "persistent_workers": self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch is not None,
        }
        self.train_loader = DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            drop_last=True,
            **self.loader_kwargs,
        )
        self.val_loader = DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=True,
            **self.loader_kwargs,
        )
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=False,
            **self.loader_kwargs,
        )

    def _evaluate_single(self, loader, title, mode, step=None):
        assert mode in ["val", "eval"]
        # re-initialize dataloader to make sure it is using the evaluation batchsize
        # (makes a difference for trainloader)
        loader = DataLoader(
            dataset=loader.dataset,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
            drop_last=False,
            **self.loader_kwargs,
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
        metrics["loss"] = torch.nn.functional.cross_entropy(
            labels_predict, labels_true
        ).item()
        labels_true, labels_predict = (
            labels_true.numpy(),
            torch.softmax(labels_predict, dim=1).numpy(),
        )

        # accuracy
        labels_predict_score = np.argmax(labels_predict, axis=1)
        metrics["accuracy"] = accuracy_score(
            labels_true, np.round(labels_predict_score)
        )
        if mode == "eval":
            LOGGER.info(f"Accuracy on {title} dataset:\t{metrics['accuracy']:.4f}")

        # auc and roc (fpr = epsB, tpr = epsS)
        metrics["auc_ovo"] = roc_auc_score(
            labels_true, labels_predict, multi_class="ovo", average="macro"
        )  # unweighted mean of AUCs across classes
        if mode == "eval":
            LOGGER.info(f"The ovo mean AUC is\t\t{metrics['auc_ovo']:.5f}")
        fpr_list, tpr_list = [], []
        for i in range(self.cfg.jc_params.num_classes):
            fpr, tpr, _ = roc_curve(labels_true == i, labels_predict[:, i])
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        # 1/epsB at fixed epsS
        def get_rej(epsS, class_idx):
            idx = np.argmin(np.abs(tpr_list[class_idx] - epsS))
            return 1 / fpr_list[class_idx][idx]

        # turn the below dict into a list please
        class_rej_dict = [None, 0.5, 0.5, 0.5, 0.5, 0.99, 0.5, 0.995, 0.5, 0.5]

        for i, rej in enumerate(class_rej_dict):
            if rej is None:
                continue
            rej_string = str(rej).replace(".", "")
            metrics[f"rej{rej_string}_{i}"] = get_rej(rej, i)
            if mode == "eval":
                LOGGER.info(
                    f"Rejection rate for class {self.class_names[i]:>10} on {title} dataset:{metrics[f'rej{rej_string}_{i}']:>5.0f} (epsS={rej})"
                )

        # create latex string
        tex_string = f"{self.cfg.run_name} & {metrics['accuracy']:.3f} & {metrics['auc_ovo']:.3f}"
        for i, rej in enumerate(class_rej_dict):
            if rej is None:
                continue
            rej_string = str(rej).replace(".", "")
            tex_string += f" & {metrics[f'rej{rej_string}_{i}']:.0f}"
        tex_string += r" \\"
        LOGGER.info(tex_string)

        if self.cfg.use_mlflow:
            for key, value in metrics.items():
                if key in ["labels_true", "labels_predict"]:
                    # do not log matrices
                    continue
                name = f"{mode}.{title}" if mode == "eval" else "val"
                log_mlflow(f"{name}.{key}", value, step=step)

        return metrics

    def _get_ypred_and_label(self, batch):
        fourmomenta = batch[0]["pf_vectors"].to(self.device)
        if self.cfg.data.features == "fourmomenta":
            scalars = torch.empty(
                fourmomenta.shape[0],
                0,
                fourmomenta.shape[2],
                device=fourmomenta.device,
                dtype=fourmomenta.dtype,
            )
        else:
            scalars = batch[0]["pf_features"].to(self.device)
        label = batch[1]["_label_"].to(self.device)
        fourmomenta, scalars, ptr = dense_to_sparse_jet(fourmomenta, scalars)
        embedding = embed_tagging_data_into_ga(fourmomenta, scalars, ptr, self.cfg.data)
        y_pred = self.model(embedding)
        if self.cfg.data.score_token:
            y_pred = y_pred.reshape(
                y_pred.shape[0] // self.cfg.data.num_global_tokens,
                self.cfg.data.num_global_tokens,
            )
        return y_pred, label
