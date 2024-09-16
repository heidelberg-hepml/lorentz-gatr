import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.tagging.experiment import TopTaggingExperiment, QGTaggingExperiment
from experiments.eventgen.processes import (
    ttbarExperiment,
    zmumuExperiment,
    z5gExperiment,
)
from experiments.tagging.jetclassexperiment import JetClassTaggingExperiment
from experiments.tagging.finetuneexperiment import TopTaggingFineTuneExperiment


@hydra.main(config_path="config", config_name="amplitudes", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "qgtagging":
        exp = QGTaggingExperiment(cfg)
    elif cfg.exp_type == "jctagging":
        exp = JetClassTaggingExperiment(cfg)
    elif cfg.exp_type == "toptaggingft":
        exp = TopTaggingFineTuneExperiment(cfg)
    elif cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    elif cfg.exp_type == "zmumu":
        exp = zmumuExperiment(cfg)
    elif cfg.exp_type == "z5g":
        exp = z5gExperiment(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()
