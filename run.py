import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.toptagging.experiment import TopTaggingExperiment

@hydra.main(config_path="config", config_name="toptagging", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    
    exp()

if __name__ == "__main__":
    main()
