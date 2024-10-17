import os
from omegaconf import OmegaConf, open_dict

from experiments.tagging.experiment import TopTaggingExperiment
from gatr.layers.linear import EquiLinear
from experiments.tagging.embedding import embed_tagging_data_into_ga


class TransferTest(TopTaggingExperiment):
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
            self.cfg.model.mean_aggregation = warmstart_cfg.model.mean_aggregation
            self.cfg.data.beam_reference = warmstart_cfg.data.beam_reference
            self.cfg.data.two_beams = warmstart_cfg.data.two_beams
            self.cfg.data.beam_token = warmstart_cfg.data.beam_token
            self.cfg.data.add_time_reference = warmstart_cfg.data.add_time_reference
            self.cfg.data.add_scalar_features = warmstart_cfg.data.add_scalar_features
            self.cfg.data.reinsert_channels = warmstart_cfg.data.reinsert_channels
            self.cfg.data.rescale_data = warmstart_cfg.data.rescale_data

            self.cfg.train = False

    def _get_ypred_and_label(self, batch):
        batch = batch.to(self.device)
        embedding = embed_tagging_data_into_ga(
            batch.x, batch.scalars, batch.ptr, self.cfg.data
        )
        y_pred_full = self.model(embedding)
        y_pred_top, y_pred_qcd = y_pred_full[:, 6], y_pred_full[:, 0]
        # require exp(y_pred)/(1+exp(y_pred)) = softmax(y_pred_top) / (softmax(y_pred_top) + softmax(y_pred_qcd))
        # solved by follow ing relation:
        y_pred = y_pred_top - y_pred_qcd
        return y_pred, batch.label.to(self.dtype)
