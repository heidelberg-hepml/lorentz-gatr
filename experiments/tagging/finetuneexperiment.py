import os, torch
from omegaconf import OmegaConf, open_dict
from torch_ema import ExponentialMovingAverage

from experiments.tagging.experiment import TopTaggingExperiment
from experiments.logger import LOGGER
from gatr.layers.linear import EquiLinear


class TopTaggingFineTuneExperiment(TopTaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load warm_start cfg
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type in ["jctagging", "binaryjetclass"]
        assert self.warmstart_cfg.data.features == "fourmomenta"

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

    def init_model(self):
        super().init_model()

        if self.warm_start:
            # nothing to do
            return

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

        # overwrite output layer
        with open_dict(self.cfg):
            self.cfg.model.net.out_s_channels = 1
        self.model.net.linear_out = EquiLinear(
            self.cfg.model.net.hidden_mv_channels,
            self.cfg.model.net.out_mv_channels,
            in_s_channels=self.cfg.model.net.hidden_s_channels,
            out_s_channels=self.cfg.model.net.out_s_channels,
        ).to(self.device)

        if self.cfg.ema:
            LOGGER.info(f"Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

    def _init_optimizer(self):
        # collect parameter lists
        params_backbone = list(self.model.net.linear_in.parameters()) + list(
            self.model.net.blocks.parameters()
        )
        params_head = self.model.net.linear_out.parameters()

        # assign parameter-specific learning rates
        param_groups = [
            {"params": params_backbone, "lr": self.cfg.finetune.lr_backbone},
            {"params": params_head, "lr": self.cfg.finetune.lr_head},
        ]

        super()._init_optimizer(param_groups=param_groups)
