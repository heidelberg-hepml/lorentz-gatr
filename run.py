import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.toptagging.experiment import TopTaggingExperiment
from experiments.eventgen.ttbarexperiment import ttbarExperiment

@hydra.main(config_path="config", config_name="ttbar", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    
    exp()

if __name__ == "__main__":
    main()
