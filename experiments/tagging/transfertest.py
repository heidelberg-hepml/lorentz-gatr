import os, time
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from experiments.tagging.experiment import TopTaggingExperiment
from gatr.layers.linear import EquiLinear
from experiments.tagging.embedding import (
    dense_to_sparse_jet,
    embed_tagging_data_into_ga,
)
from experiments.logger import LOGGER

from experiments.tagging.miniweaver.dataset import SimpleIterDataset
from experiments.tagging.miniweaver.loader import to_filelist


class TopTransferTest(TopTaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load warm_start cfg
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type == "jctagging"
        assert self.warmstart_cfg.data.features in [
            "fourmomenta",
            "fourmomenta_extended",
        ]
        if self.warmstart_cfg.data.score_token:
            raise NotImplementedError(
                "Score-token option not properly implemented yet to be transferred from jc to top"
            )

        # merge config files
        with open_dict(self.cfg):
            # overwrite model
            self.cfg.model = self.warmstart_cfg.model
            self.cfg.ema = self.warmstart_cfg.ema
            self.cfg.ga_representations = self.warmstart_cfg.ga_representations

            # overwrite model-specific data entries
            self.cfg.model.mean_aggregation = self.warmstart_cfg.model.mean_aggregation
            self.cfg.data.beam_reference = self.warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = self.warmstart_cfg.data.two_beams
            self.cfg.data.beam_token = self.warmstart_cfg.data.beam_token
            self.cfg.data.add_time_reference = (
                self.warmstart_cfg.data.add_time_reference
            )
            self.cfg.data.add_xzplane = self.warmstart_cfg.data.add_xzplane
            self.cfg.data.add_yzplane = self.warmstart_cfg.data.add_yzplane
            self.cfg.data.add_scalar_features = (
                self.warmstart_cfg.data.add_scalar_features
            )
            self.cfg.data.reinsert_channels = self.warmstart_cfg.data.reinsert_channels
            self.cfg.data.rescale_data = self.warmstart_cfg.data.rescale_data
            self.cfg.data.scalar_features_preprocessing = (
                self.warmstart_cfg.data.scalar_features_preprocessing
            )

            self.cfg.train = False

    def init_model(self):
        super().init_model()

        # load pretrained weights
        model_path = os.path.join(
            self.warmstart_cfg.run_dir,
            "models",
            f"model_run{self.warmstart_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu")["model"]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        embedding = embed_tagging_data_into_ga(
            batch.x, batch.scalars, batch.ptr, self.cfg.data
        )
        y_pred_full = self.model(embedding)
        y_pred_top, y_pred_qcd = y_pred_full[:, 6], y_pred_full[:, 0]
        # require exp(y_pred)/(1+exp(y_pred)) = softmax(y_pred_top) / (softmax(y_pred_top) + softmax(y_pred_qcd))
        # solved by following relation:
        y_pred = y_pred_top - y_pred_qcd
        return y_pred, batch.label.to(self.dtype)


class JetClassTransferTest(TopTaggingExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = [
            "ZJetsToNuNu",
            "TTBar",
        ]
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type == "jctagging"
        assert self.warmstart_cfg.data.features == "fourmomenta"
        if self.warmstart_cfg.data.score_token:
            raise NotImplementedError(
                "Score-token option not properly implemented yet to be transferred from jc to top"
            )

        with open_dict(self.cfg):
            # overwrite model
            self.cfg.model = self.warmstart_cfg.model
            self.cfg.ema = self.warmstart_cfg.ema
            self.cfg.ga_representations = self.warmstart_cfg.ga_representations

            # overwrite model-specific data entries
            self.cfg.model.mean_aggregation = self.warmstart_cfg.model.mean_aggregation
            self.cfg.data.beam_reference = self.warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = self.warmstart_cfg.data.two_beams
            self.cfg.data.beam_token = self.warmstart_cfg.data.beam_token
            self.cfg.data.add_time_reference = (
                self.warmstart_cfg.data.add_time_reference
            )
            self.cfg.data.add_xzplane = self.warmstart_cfg.data.add_xzplane
            self.cfg.data.add_yzplane = self.warmstart_cfg.data.add_yzplane
            self.cfg.data.add_scalar_features = (
                self.warmstart_cfg.data.add_scalar_features
            )
            self.cfg.data.reinsert_channels = self.warmstart_cfg.data.reinsert_channels
            self.cfg.data.rescale_data = self.warmstart_cfg.data.rescale_data
            self.cfg.data.scalar_features_preprocessing = (
                self.warmstart_cfg.data.scalar_features_preprocessing
            )
            self.cfg.data.features = self.warmstart_cfg.data.features

            self.cfg.jc_params = self.warmstart_cfg.jc_params
            self.cfg.train = False
            self.cfg.evaluation.eval_set = ["test"]

            if self.warmstart_cfg.data.features == "fourmomenta":
                self.cfg.data.data_config = (
                    "experiments/tagging/miniweaver/fourmomenta_jctt.yaml"
                )
            else:
                raise ValueError(
                    f"warmstart with data.features={self.warmstart_cfg.data.features} not supported"
                )

    def init_data(self):
        LOGGER.info(f"Creating SimpleIterDataset")
        t0 = time.time()

        datasets = {"test": None}

        for_training = {"test": False}
        folder = {"test": "test_20M"}
        files_range = {
            "test": self.cfg.data.test_files_range,
        }
        self.num_files = {
            label: frange[1] - frange[0] for label, frange in files_range.items()
        }
        for label in ["test"]:
            path = os.path.join(self.cfg.data.data_dir, folder[label])
            flist = [
                f"{path}/{classname}_{str(i).zfill(3)}.root"
                for classname in self.class_names
                for i in range(*files_range[label])
            ]
            file_dict, files = to_filelist(flist)

            LOGGER.info(f"Using {len(flist)} files for {label}ing from {path}")
            datasets[label] = SimpleIterDataset(
                file_dict,
                self.cfg.data.data_config,
                for_training=for_training[label],
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
        self.data_test = datasets["test"]

        dt = time.time() - t0
        LOGGER.info(f"Finished creating datasets after {dt:.2f} s = {dt/60:.2f} min")

    def _init_dataloader(self):
        self.loader_kwargs = {
            "pin_memory": True,
            "persistent_workers": self.cfg.jc_params.num_workers > 0
            and self.cfg.jc_params.steps_per_epoch is not None,
        }
        num_workers = {
            label: min(self.cfg.jc_params.num_workers, self.num_files[label])
            for label in ["test"]
        }
        self.test_loader = DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.evaluation.batchsize,
            drop_last=False,
            num_workers=num_workers["test"],
            **self.loader_kwargs,
        )
        self.train_loader, self.val_loader = None, None

    def init_model(self):
        super().init_model()

        # load pretrained weights
        model_path = os.path.join(
            self.warmstart_cfg.run_dir,
            "models",
            f"model_run{self.warmstart_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu")["model"]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

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
        y_pred_full = self.model(embedding)

        y_pred_top, y_pred_qcd = y_pred_full[:, 6], y_pred_full[:, 0]
        # require exp(y_pred)/(1+exp(y_pred)) = softmax(y_pred_top) / (softmax(y_pred_top) + softmax(y_pred_qcd))
        # solved by following relation:
        y_pred = y_pred_top - y_pred_qcd
        # note: label 1 for top and 0 for qcd -> no changes necessary

        return y_pred, label
