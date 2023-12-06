# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

import numpy as np
import torch
import yaml

class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights."""


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_config(path):
    """
    Method to load a parameter dict from a yaml file
    :param path: path to a *.yaml parameter file
    :return: the parameters as a dict
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def save_config(config, name="config.yaml"):
    """
    Method to save a parameter dict to a yaml file
    :param params: the parameter dict
    :param name: the name of the yaml file
    """
    with open(name, 'w') as f:
        yaml.dump(config, f)

def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(-1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :])
