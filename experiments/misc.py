import torch
import yaml


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def save_config(config, name="config.yaml"):
    with open(name, 'w') as f:
        yaml.dump(config, f)
