import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid

import os, time
from experiments.logger import LOGGER
from experiments.eventgen.helpers import (
    fourmomenta_to_jetmomenta,
    delta_r_fast,
    get_virtual_particle,
)
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve


class MLPClassifier:
    def __init__(self, net, cfg_training, experiment, device):
        self.net = net
        self.cfg_training = cfg_training
        self.exp = experiment  # this is bad style (but convenient)
        self.device = device

    def preprocess(self, events, cls_params, eps=1e-10):
        # naive channels
        naive_raw = fourmomenta_to_jetmomenta(events)
        naive = naive_raw.reshape(
            *naive_raw.shape[:-2], naive_raw.shape[-2] * naive_raw.shape[-1]
        )

        # delta_r channels
        dr = delta_r_fast(naive_raw[..., None, :], naive_raw[..., None, :, :])
        rows, cols = torch.triu_indices(dr.shape[-1], dr.shape[-1], offset=1)
        dr = dr[:, rows, cols]  # extract only upper triangular part

        # mass of virtual particles
        virtual = []
        for idx in self.exp.virtual_components:
            v = get_virtual_particle(naive_raw, idx)
            virtual.append(v)
        virtual = torch.stack(virtual, dim=-2)
        virtual = virtual.reshape(
            *virtual.shape[:-2], virtual.shape[-2] * virtual.shape[-1]
        )

        # combine everything and standardize
        x = torch.cat((naive, dr, virtual), dim=-1)
        if cls_params["mean"] is None or cls_params["std"] is None:
            cls_params["mean"] = x.mean(dim=0, keepdim=True)
            cls_params["std"] = x.std(dim=0, keepdim=True)
            cls_params["std"][
                cls_params["std"] < 1e-3
            ] = 1.0  # regularize (only relevant for dR diagonal)
        x = (x - cls_params["mean"]) / cls_params["std"]
        assert torch.isfinite(x).all()
        return x, cls_params

    def init_training(self):
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.cfg_training.lr
        )

        if self.cfg_training.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.cfg_training.lr_factor,
                patience=self.cfg_training.lr_patience,
            )
        else:
            self.scheduler = None

        self.loss = nn.BCEWithLogitsLoss()
        self.get_LR = lambda x: torch.exp(x)  # likelihood ratio function

    def train_test_val_split(self, data):
        splits = np.round(
            np.cumsum(self.cfg_training.train_test_val) * data.shape[0]
        ).astype("int")
        trn, tst, val, _ = np.split(data, splits, axis=0)
        return {"trn": trn, "tst": tst, "val": val}

    def init_data(self, data_true, data_fake):
        def create_dataloader(x, shuffle=True):
            return torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x),
                batch_size=self.cfg_training.batchsize,
                shuffle=shuffle,
            )

        self.loaders_trn = [create_dataloader(x["trn"]) for x in [data_true, data_fake]]
        self.loaders_val = [create_dataloader(x["val"]) for x in [data_true, data_fake]]
        self.loaders_tst = [create_dataloader(x["tst"]) for x in [data_true, data_fake]]

    def init_dataloaders(self, data_true, data_fake):
        # data_true and data_fake have shape (nevents, ncomponents)
        LOGGER.info(
            f"Classifier true/fake data has shape {data_true.shape}/{data_fake.shape}"
        )

        # train/val split
        n_true, n_fake = data_true.shape[0], data_fake.shape[0]
        split_true = [int(n_true * ratio) for ratio in self.cfg_training.train_val]
        split_fake = [int(n_fake * ratio) for ratio in self.cfg_training.train_val]
        data_trn = data_true[: split_true[0]], data_fake[: split_fake[0]]
        data_val = (
            data_true[split_true[0] : sum(split_true)],
            data_fake[split_fake[0] : sum(split_fake)],
        )

        # initialize loaders
        self.loaders_trn = [self.create_dataloader(x) for x in data_trn]
        self.loaders_val = [self.create_dataloader(x) for x in data_val]

    @torch.set_grad_enabled(True)
    def train(self):
        self.init_training()
        self.tracker = {"loss": [], "val_loss": [], "lr": []}

        LOGGER.info(
            f"Starting to train classifier for {self.cfg_training.nepochs} epochs"
        )
        t0 = time.time()
        smallest_val_loss, smallest_val_loss_epoch, es_patience = 1e10, 0, 0
        for epoch in range(self.cfg_training.nepochs):
            for (x_true,), (x_fake,) in zip(*self.loaders_trn):
                self.training_step(x_true, x_fake)

            val_loss = self.validate()
            if self.cfg_training.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(val_loss)
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                smallest_val_loss_epoch = epoch
                es_patience = 0

                if self.cfg_training.load_best_model and self.exp.cfg.save:
                    path = os.path.join(
                        self.exp.cfg.run_dir,
                        "models",
                        f"classifier_{self.exp.cfg.run_idx}_epoch{epoch}.pt",
                    )
                    torch.save(self.net.state_dict(), path)
            else:
                es_patience += 1
                if es_patience > self.cfg_training.es_patience:
                    LOGGER.info(f"Early stopping classifier in epoch {epoch}")
                    break

        dt = time.time() - t0
        LOGGER.info(f"Finished classifier training after {dt/60:.2f} min")
        if self.cfg_training.load_best_model and self.exp.cfg.save:
            path = os.path.join(
                self.exp.cfg.run_dir,
                "models",
                f"classifier_{self.exp.cfg.run_idx}_epoch{smallest_val_loss_epoch}.pt",
            )
            state_dict = torch.load(path, map_location=self.device)
            LOGGER.info(f"Loading model from {path}")
            self.net.load_state_dict(state_dict)

    def training_step(self, x_true, x_fake):
        self.net.train()
        loss = self.batch_loss(x_true, x_fake)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.tracker["loss"].append(loss.detach().item())
        self.tracker["lr"].append(self.optimizer.param_groups[0]["lr"])

    @torch.no_grad()
    def validate(self):
        self.net.eval()
        losses = []
        for (x_true,), (x_fake,) in zip(*self.loaders_val):
            loss = self.batch_loss(x_true, x_fake)
            losses.append(loss)
        val_loss = torch.stack(losses, dim=0).mean()
        self.tracker["val_loss"].append(val_loss)
        return val_loss

    def batch_loss(self, x_true, x_fake):
        x_true, x_fake = x_true.to(self.device), x_fake.to(self.device)
        x = torch.cat((x_true, x_fake), dim=0)
        labels = torch.cat(
            (torch.ones_like(x_true[:, [0]]), torch.zeros_like(x_fake[:, [0]])), dim=0
        )
        logits = self.net(x)
        loss = self.loss(logits, labels)
        return loss

    def evaluate(self):
        # evaluate model on tst set
        self.net.eval()
        scores_true, scores_fake = [], []
        for (x,) in self.loaders_tst[0]:
            scores_true.append(self.net(x.to(self.device)))
        for (x,) in self.loaders_tst[1]:
            scores_fake.append(self.net(x.to(self.device)))
        scores_true = torch.cat(scores_true, dim=0).squeeze()
        scores_fake = torch.cat(scores_fake, dim=0).squeeze()
        labels = torch.cat(
            (torch.ones_like(scores_true), torch.zeros_like(scores_fake)), dim=0
        )
        scores = torch.cat((scores_true, scores_fake), dim=0)  # raw network output
        logits = sigmoid(scores)  # probabilities
        weights = self.get_LR(scores)  # likelihood ratio p_true/p_fake

        # extract metrics
        fpr, tpr, th = roc_curve(labels, logits)
        auc = roc_auc_score(labels, logits)
        accuracy = accuracy_score(labels, torch.round(logits))
        prob_true, prob_pred = calibration_curve(labels, logits, n_bins=30)
        LOGGER.info(f"Classifier score: AUC={auc:.4f}, accuracy={accuracy:.4f}")
        self.results = {
            "labels": {
                "all": labels,
                "true": torch.ones_like(scores_true),
                "fake": torch.zeros_like(scores_fake),
            },
            "scores": {"all": scores, "true": scores_true, "fake": scores_fake},
            "logits": {
                "all": logits,
                "true": sigmoid(scores_true),
                "fake": sigmoid(scores_fake),
            },
            "weights": {
                "all": weights,
                "true": self.get_LR(scores_true),
                "fake": self.get_LR(scores_fake),
            },
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "accuracy": accuracy,
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }
