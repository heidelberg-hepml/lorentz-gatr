import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment
from experiments.toptagging.experiment import TopTaggingExperiment
from experiments.eventgen.ttbarexperiment import ttbarExperiment
from experiments.eventgen.zmumuexperiment import zmumuExperiment

@hydra.main(config_path="config", config_name="ttbar", version_base=None)
def main(cfg):
    if cfg.exp_type == "amplitudes":
        exp = AmplitudeExperiment(cfg)
    elif cfg.exp_type == "toptagging":
        exp = TopTaggingExperiment(cfg)
    elif cfg.exp_type == "ttbar":
        exp = ttbarExperiment(cfg)
    elif cfg.exp_type == "zmumu":
        exp = zmumuExperiment(cfg)
    else:
        raise ValueError(f"exp_type={cfg.exp_tyep} not implemented")
    
    exp()

if __name__ == "__main__":
    main()
