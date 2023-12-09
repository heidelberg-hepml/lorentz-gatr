import hydra
from experiments.amplitudes.experiment import AmplitudeExperiment

@hydra.main(config_path="config", config_name="amplitudes", version_base=None)
def main(cfg):
    exp = AmplitudeExperiment(cfg)
    exp()

if __name__ == "__main__":
    main()
