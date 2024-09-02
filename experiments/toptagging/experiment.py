import numpy as np
import torch
import torch_geometric

import os, time
from omegaconf import OmegaConf, open_dict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from experiments.base_experiment import BaseExperiment
from experiments.toptagging.dataset import TopTaggingDataset
from experiments.toptagging.dataset import QGTaggingDataset
from experiments.toptagging.plots import plot_mixer
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow

import matplotlib.pyplot as plt

from data.utils.dataset import SimpleIterDataset
from data.utils.loader import to_filelist

MODEL_TITLE_DICT = {"GATr": "GATr", "Transformer": "Tr", "MLP": "MLP"}


class TaggingExperiment(BaseExperiment):
    """
    Generalization of all tagging experiments
    """

    def init_physics(self):
        if not self.cfg.training.force_xformers:
            LOGGER.warning(
                f"Using training.force_xformers=False, this will slow down the network by a factor of 5-10."
            )

        # dynamically extend dict
        with open_dict(self.cfg):
            gatr_name = ["experiments.toptagging.wrappers.TopTaggingGATrWrapper", "experiments.toptagging.wrappers.TopTaggingPretrainGATrWrapper"]
            assert self.cfg.model._target_ in gatr_name

            # global token?
            if self.cfg.model._target_ in gatr_name:
                self.cfg.data.include_global_token = not self.cfg.model.mean_aggregation
            else:
                raise ValueError(f"model {self.cfg.model._target_} not implemented")

            if self.cfg.exp_type == "toptagging":
                # make sure we know where we start from
                self.cfg.model.net.in_s_channels = 0
                self.cfg.model.net.in_mv_channels = 1
            elif self.cfg.exp_type == "qgtagging":
                # We add 7 scalar channels, 1 for the global token and 6 for the particle id features
                # (charge, electron, muon, photon, charged hadron and neutral hadron)
                self.cfg.model.net.in_s_channels = 6
                self.cfg.model.net.in_mv_channels = 1

            # extra scalar channels
            if self.cfg.data.add_pt:
                self.cfg.model.net.in_s_channels += 1
            if self.cfg.data.include_global_token:
                self.cfg.model.net.in_s_channels += 1

            # extra mv channels for beam_reference and time_reference
            if not self.cfg.data.beam_token:
                if self.cfg.data.beam_reference is not None:
                    self.cfg.model.net.in_mv_channels += (
                        2
                        if self.cfg.data.two_beams
                        and not self.cfg.data.beam_reference == "xyplane"
                        else 1
                    )
                if self.cfg.data.add_time_reference:
                    self.cfg.model.net.in_mv_channels += 1

            # reinsert channels
            if self.cfg.data.reinsert_channels:
                self.cfg.model.net.reinsert_mv_channels = list(
                    range(self.cfg.model.net.in_mv_channels)
                )
                self.cfg.model.net.reinsert_s_channels = list(
                    range(self.cfg.model.net.in_s_channels)
                )

    def init_data(self):
        raise NotImplementedError

    def _init_data(self, Dataset, data_path):
        LOGGER.info(f"Creating {Dataset.__name__} from {data_path}")
        t0 = time.time()
        kwargs = {
            "cfg": self.cfg,
            "dtype": self.dtype,
            "device": self.device,
        }
        self.data_train = Dataset(data_path, "train", data_scale=None, **kwargs)
        self.data_test = Dataset(
            data_path, "test", data_scale=self.data_train.data_scale, **kwargs
        )
        self.data_val = Dataset(
            data_path, "val", data_scale=self.data_train.data_scale, **kwargs
        )
        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_data_pretrain(self, Dataset):
        LOGGER.info(f"Creating {Dataset.__name__}")
        t0 = time.time()

        train_file_dict, self.train_files = to_filelist(self.cfg.jc_params, 'train')
        val_file_dict, self.val_files = to_filelist(self.cfg.jc_params, 'val')
        test_file_dict, self.test_files = to_filelist(self.cfg.jc_params, 'test')
        train_range = val_range = test_range = (0, 1)

        LOGGER.info(f"Using {len(self.train_files)} files for training, range: {str(train_range)}")
        LOGGER.info(f"Using {len(self.val_files)} files for validation, range: {str(val_range)}")
        LOGGER.info(f"Using {len(self.test_files)} files for validation, range: {str(test_range)}")

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

    def init_dataloader(self):
        raise NotImplementedError

    def _init_dataloader(self):
        self.train_loader = torch_geometric.loader.DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            shuffle=True,
        )
        self.test_loader = torch_geometric.loader.DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )
        self.val_loader = torch_geometric.loader.DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            shuffle=False,
        )

        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )

    def _init_dataloader_pretrain(self):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.training.batchsize,
            drop_last=True,
            pin_memory=True,
            num_workers=min(self.cfg.jc_params.num_workers, int(len(self.train_files) * self.cfg.jc_params.file_fraction)),
            persistent_workers=self.cfg.jc_params.num_workers > 0 and self.cfg.jc_params.steps_per_epoch is not None,
        )
        self.val_loader = torch.utils.data.DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=True,
            pin_memory=True,
            num_workers=min(self.cfg.jc_params.num_workers, int(len(self.val_files) * int(self.cfg.jc_params.file_fraction))),
            persistent_workers=self.cfg.jc_params.num_workers > 0 and self.cfg.jc_params.steps_per_epoch_val is not None,
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=False,
            pin_memory=True,
            num_workers=min(self.cfg.jc_params.num_workers, len(self.test_files)),
        )

        """
        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), {self.cfg.evaluation.batchsize} (evaluation)"
        )
        """

    def evaluate(self):
        self.results = {}

        # this is a bit ugly, but it does the job
        if self.ema is not None:
            with self.ema.average_parameters():
                self.results["train"] = self._evaluate_single(
                    self.train_loader, "train", mode="eval"
                )
                self.results["val"] = self._evaluate_single(
                    self.val_loader, "val", mode="eval"
                )
                self.results["test"] = self._evaluate_single(
                    self.test_loader, "test", mode="eval"
                )

            self._evaluate_single(self.train_loader, "train_noema", mode="eval")
            self._evaluate_single(self.val_loader, "val_noema", mode="eval")
            self._evaluate_single(self.test_loader, "test_noema", mode="eval")

        else:
            self.results["train"] = self._evaluate_single(
                self.train_loader, "train", mode="eval"
            )
            self.results["val"] = self._evaluate_single(
                self.val_loader, "val", mode="eval"
            )
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

        #if mode == "eval":
        #    LOGGER.info(
        #        f"### Starting to evaluate model on {title} dataset with "
        #        f"{len(loader.dataset.data_list)} elements, batchsize {loader.batch_size} ###"
        #    )
        metrics = {}

        # predictions
        labels_true, labels_predict = [], []
        self.model.eval()
        if self.cfg.training.optimizer == "ScheduleFree":
            self.optimizer.eval()
        with torch.no_grad():
            if self.cfg.exp_type == "jctagging":
                for batch in loader:
                    input = batch[0]['pf_vectors'].to(torch.device("cuda"))
                    label = batch[1]['_label_'].to(torch.device("cuda"))
                    y_pred = self.model(input).reshape(-1, 10)
                    y_pred = torch.nn.functional.sigmoid(y_pred)
                    labels_true.append(label.cpu())
                    labels_predict.append(y_pred.cpu().float())
            else:
                for batch in loader:
                    batch = batch.to(self.device)
                    y_pred = self.model(batch)
                    y_pred = torch.nn.functional.sigmoid(y_pred)
                    labels_true.append(batch.label.cpu().float())
                    labels_predict.append(y_pred.cpu().float())

        labels_true, labels_predict = torch.cat(labels_true), torch.cat(labels_predict)
        if mode == "eval":
            metrics["labels_true"], metrics["labels_predict"] = (
                labels_true,
                labels_predict,
            )


        if self.cfg.exp_type == "jctagging":
            # bce loss
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

        else:
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

        if self.cfg.evaluate and self.cfg.evaluation.save_roc:
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

    def _init_loss_pretrain(self):
        self.loss = torch.nn.CrossEntropyLoss()

    # overwrite _validate method to compute metrics over the full validation set
    def _validate(self, step):
        if self.ema is not None:
            with self.ema.average_parameters():
                metrics = self._evaluate_single(
                    self.val_loader, "val", mode="val", step=step
                )
        else:
            metrics = self._evaluate_single(
                self.val_loader, "val", mode="val", step=step
            )

        if self.cfg.exp_type == "jctagging":
            self.val_loss.append(metrics["ce"])
            return metrics["ce"]
        else:
            self.val_loss.append(metrics["bce"])
            return metrics["bce"]

    def _batch_loss(self, batch):
        if self.cfg.exp_type == "jctagging":
            input = batch[0]['pf_vectors'].to(torch.device("cuda"))
            label = batch[1]['_label_'].to(torch.device("cuda"))
            y_pred = self.model(input).reshape(-1,10)
            loss = self.loss(y_pred, label)
        elif self.cfg.exp_type == "toptagging" or self.cfg.exp_type == "qgtagging":
            y_pred = self.model(batch)
            loss = self.loss(y_pred, batch.label.to(self.dtype))
        assert torch.isfinite(loss).all()

        metrics = {}
        return loss, metrics

    def _init_metrics(self):
        return {}


class TopTaggingExperiment(TaggingExperiment):
    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"toptagging_{self.cfg.data.dataset}.npz"
        )
        self._init_data(TopTaggingDataset, data_path)
    def init_dataloader(self):
        self._init_dataloader()
    def init_loss(self):
        self._init_loss()

class QGTaggingExperiment(TaggingExperiment):
    def init_data(self):
        data_path = os.path.join(
            self.cfg.data.data_dir, f"qg_tagging_{self.cfg.data.dataset}.npz"
        )
        self._init_data(QGTaggingDataset, data_path)
    def init_dataloader(self):
        self._init_dataloader()
    def init_loss(self):
        self._init_loss()

class JetClassTaggingExperiment(TaggingExperiment):
    def init_data(self):
        self._init_data_pretrain(SimpleIterDataset)
    def init_dataloader(self):
        self._init_dataloader_pretrain()
    def init_loss(self):
        self._init_loss_pretrain()
