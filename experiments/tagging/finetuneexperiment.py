import os
from omegaconf import OmegaConf, open_dict

from experiments.tagging.experiment import TopTaggingExperiment
from gatr.layers.linear import EquiLinear


class TopTaggingFineTuneExperiment(TopTaggingExperiment):
    def __init__(self, cfg):
        super().__init__(cfg)

        # load warm_start cfg
        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        warmstart_cfg = OmegaConf.load(warmstart_path)
        assert warmstart_cfg.exp_type == "jctagging"
        assert warmstart_cfg.data.features == "fourmomenta"
        assert (
            warmstart_cfg.ema
            and self.cfg.ema
            or not warmstart_cfg.ema
            and not self.cfg.ema
        ), "Current implementation only works if pretrained and finetune model use the same EMA setting"
        if warmstart_cfg.data.score_token:
            raise NotImplementedError(
                "Score-token option not properly implemented yet to be transferred from jc to top"
            )

        # merge config files
        with open_dict(self.cfg):
            # overwrite model
            self.cfg.model = warmstart_cfg.model

            # overwrite model-specific data entries
            # (this is ugly, please improve it)
            self.cfg.data.beam_reference = warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = warmstart_cfg.data.two_beams
            self.cfg.data.beam_token = warmstart_cfg.data.beam_token
            self.cfg.data.add_time_reference = warmstart_cfg.data.add_time_reference
            self.cfg.data.add_pt = warmstart_cfg.data.add_pt
            self.cfg.data.add_energy = warmstart_cfg.data.add_energy
            self.cfg.data.reinsert_channels = warmstart_cfg.data.reinsert_channels
            self.cfg.data.rescale_data = warmstart_cfg.data.rescale_data

    def init_model(self):
        super().init_model()

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
