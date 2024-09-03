import numpy as np
import torch
from torch.utils.data import DataLoader

import time

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

from experiments.toptagging.experiment import TaggingExperiment

from data.utils.dataset import SimpleIterDataset
from data.utils.loader import to_filelist

def init_data_pretrain(self, Dataset):
    LOGGER.info(f"Creating {Dataset.__name__}")
    t0 = time.time()

    train_file_dict, self.train_files = to_filelist(self.cfg.jc_params, 'train')
    val_file_dict, self.val_files = to_filelist(self.cfg.jc_params, 'val')
    test_file_dict, self.test_files = to_filelist(self.cfg.jc_params, 'test')
    train_range = val_range = test_range = (0, 1)

    LOGGER.info(f"Using {len(self.train_files)} files for training, range: {str(train_range)}")
    LOGGER.info(f"Using {len(self.val_files)} files for validation, range: {str(val_range)}")
    LOGGER.info(f"Using {len(self.test_files)} files for testing, range: {str(test_range)}")

    self.data_train = Dataset(train_file_dict, self.cfg.jc_params.data_config, for_training=True,
        extra_selection=self.cfg.jc_params.extra_selection,
        remake_weights=not self.cfg.jc_params.not_remake_weights,
        load_range_and_fraction=(train_range, self.cfg.jc_params.data_fraction),
        file_fraction=self.cfg.jc_params.file_fraction,
        fetch_by_files=self.cfg.jc_params.fetch_by_files,
        fetch_step=self.cfg.jc_params.fetch_step,
        infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
        in_memory=self.cfg.jc_params.in_memory,
        name='train')

    self.data_val = Dataset(val_file_dict, self.cfg.jc_params.data_config, for_training=True,
        extra_selection=self.cfg.jc_params.extra_selection,
        remake_weights=not self.cfg.jc_params.not_remake_weights,
        load_range_and_fraction=(val_range, self.cfg.jc_params.data_fraction),
        file_fraction=self.cfg.jc_params.file_fraction,
        fetch_by_files=self.cfg.jc_params.fetch_by_files,
        fetch_step=self.cfg.jc_params.fetch_step,
        infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
        in_memory=self.cfg.jc_params.in_memory,
        name='val')

    self.data_test = Dataset(test_file_dict, self.cfg.jc_params.data_config, for_training=False,
        extra_selection=self.cfg.jc_params.extra_selection,
        remake_weights=not self.cfg.jc_params.not_remake_weights,
        load_range_and_fraction=(test_range, self.cfg.jc_params.data_fraction),
        file_fraction=self.cfg.jc_params.file_fraction,
        fetch_by_files=self.cfg.jc_params.fetch_by_files,
        fetch_step=self.cfg.jc_params.fetch_step,
        infinity_mode=self.cfg.jc_params.steps_per_epoch is not None,
        in_memory=self.cfg.jc_params.in_memory,
        name='test')

    dt = time.time() - t0
    LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")


def init_dataloader_pretrain(self):
    self.train_loader = DataLoader(
        dataset=self.data_train,
        batch_size=self.cfg.training.batchsize,
        drop_last=True,
        pin_memory=True,
        num_workers=min(self.cfg.jc_params.num_workers, int(len(self.train_files) * self.cfg.jc_params.file_fraction)),
        persistent_workers=self.cfg.jc_params.num_workers > 0 and self.cfg.jc_params.steps_per_epoch is not None,
    )
    self.val_loader = DataLoader(
        dataset=self.data_val,
        batch_size=self.cfg.evaluation.batchsize,
        drop_last=True,
        pin_memory=True,
        num_workers=min(self.cfg.jc_params.num_workers, int(len(self.val_files) * int(self.cfg.jc_params.file_fraction))),
        persistent_workers=self.cfg.jc_params.num_workers > 0 and self.cfg.jc_params.steps_per_epoch_val is not None,
    )
    self.test_loader = DataLoader(
        dataset=self.data_test,
        batch_size=self.cfg.evaluation.batchsize,
        drop_last=False,
        pin_memory=True,
        num_workers=min(self.cfg.jc_params.num_workers, len(self.test_files)),
    )

def evaluate_pretrain(self):
    self.results = {}

    if self.ema is not None:
        with self.ema.average_parameters():
            self.results["train"] = self.evaluate_single_pretrain(
                self.train_loader, "train", mode="eval"
            )
            self.results["val"] = self.evaluate_single_pretrain(
                self.train_loader, "val", mode="eval"
            )
            self.results["test"] = self.evaluate_single_pretrain(
                self.train_loader, "test", mode="eval"
            )

        evaluate_single_pretrain(self, self.train_loader, "train_noema", mode="eval")
        evaluate_single_pretrain(self, self.val_loader, "val_noema", mode="eval")
        evaluate_single_pretrain(self, self.test_loader, "test_noema", mode="eval")

    else:
        self.results["train"] = evaluate_single_pretrain(
            self, self.train_loader, "train", mode="eval"
        )
        self.results["val"] = evaluate_single_pretrain(
            self, self.train_loader, "val", mode="eval"
        )
        self.results["test"] = evaluate_single_pretrain(
            self, self.train_loader, "test", mode="eval"
        )



def evaluate_single_pretrain(self, loader, title, mode, step=None):
    assert mode in ["val", "eval"]
    # re-initialize dataloader to make sure it is using the evaluation batchsize (makes a difference for trainloader)
    loader = DataLoader(
        dataset=loader.dataset,
        batch_size=self.cfg.evaluation.batchsize,
        shuffle=False,
    )

    metrics = {}

    # predictions
    labels_true, labels_predict = [], []
    self.model.eval()
    if self.cfg.training.optimizer == "ScheduleFree":
        self.optimizer.eval()
    with torch.no_grad():
        for batch in loader:
            input = batch[0]['pf_vectors'].to(torch.device("cuda"))
            label = batch[1]['_label_'].to(torch.device("cuda"))
            y_pred = self.model(input).reshape(-1, 10)
            labels_true.append(label.cpu())
            labels_predict.append(y_pred.cpu().float())

    labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
    if mode == "eval":
        metrics["labels_true"], metrics["labels_predict"] = (
            labels_true,
            labels_predict,
        )

    # ce loss
    metrics["ce"] = torch.nn.functional.cross_entropy(
        labels_predict, labels_true
    ).item()
    labels_true, labels_predict = labels_true.numpy(), torch.softmax(labels_predict, dim=1).numpy()

    # accuracy
    labels_predict_score = np.argmax(labels_predict, axis=1)
    metrics["accuracy"] = accuracy_score(labels_true, np.round(labels_predict_score))
    if mode == "eval":
        LOGGER.info(f"Accuracy on {title} dataset: {metrics['accuracy']:.4f}")

    # roc (fpr = epsB, tpr = epsS)
    fpr_list, tpr_list, auc_scores = [], [], []
    for i in range(10):
        fpr, tpr, _ = roc_curve(labels_true == i, labels_predict[:, i])
        auc_score = roc_auc_score(labels_true == i, labels_predict[:, i])
        auc_scores.append(auc_score)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        metrics["auc_class_{}".format(i)] = auc_score
        if mode == "eval":
            LOGGER.info(f"AUC score for class {i} on {title} dataset: {auc_score:.4f}")

    metrics["auc_total"] = np.mean(auc_scores)

    # 1/epsB at fixed epsS
    def get_rej(epsS, class_idx):
        idx = np.argmin(np.abs(tpr_list[class_idx] - epsS))
        return 1 / fpr_list[class_idx][idx]

    for i in range(10):
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

def init_loss_pretrain(self):
    self.loss = torch.nn.CrossEntropyLoss()

def validate_pretrain(self, step):
    if self.ema is not None:
        with self.ema.average_parameters():
            metrics = self._evaluate_single(
                self.val_loader, "val", mode="val", step=step
            )
    else:
        metrics = self._evaluate_single(
            self.val_loader, "val", mode="val", step=step
        )

    self.val_loss.append(metrics["ce"])
    return metrics["ce"]

def batch_loss_pretrain(self, batch):
    input = batch[0]['pf_vectors'].to(torch.device("cuda"))
    label = batch[1]['_label_'].to(torch.device("cuda"))
    y_pred = self.model(input).reshape(-1,10)
    loss = self.loss(y_pred, label)
    assert torch.isfinite(loss).all()

    metrics = {}
    return loss, metrics


class JetClassTaggingExperiment(TaggingExperiment):
    def init_data(self):
        init_data_pretrain(self, SimpleIterDataset)
    def init_dataloader(self):
        init_dataloader_pretrain(self)
    def init_loss(self):
        init_loss_pretrain(self)
    def _batch_loss(self, batch):
        return batch_loss_pretrain(self, batch)
    def _validate(self, step):
        validate_pretrain(self, step)
    def evaluate(self):
        evaluate_pretrain(self)
