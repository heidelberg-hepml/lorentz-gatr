import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.toptagging.experiment import TopTaggingExperiment
from experiments.eventgen.processes import (
    ttbarExperiment,
    zmumuExperiment,
    z5gExperiment,
)
from experiments.toptagging.experiment import QGTaggingExperiment
from experiments.toptagging.experiment import JetClassTaggingExperiment

@hydra.main(config_path="config", config_name="jctagging", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "qgtagging":
        exp = QGTaggingExperiment(cfg)
    elif cfg.exp_type == "jctagging":
        exp = JetClassTaggingExperiment(cfg)

    exp()


if __name__ == "__main__":
    main()
