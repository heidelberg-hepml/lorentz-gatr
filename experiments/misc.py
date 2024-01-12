# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

from collections.abc import Mapping
import numpy as np
import torch

class NaNError(BaseException):
    """Exception to be raise when the training encounters a NaN in loss or model weights."""


def get_device() -> torch.device:
    """Gets CUDA if available, CPU else."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_nd(tensor, d):
    """Make tensor n-dimensional, group extra dimensions in first."""
    return tensor.view(-1, *(1,) * (max(0, d - 1 - tensor.dim())), *tensor.shape[-(d - 1) :])

def flatten_dict(d, parent_key="", sep="."):
    """Flattens a nested dictionary with str keys."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def frequency_check(step, every_n_steps, skip_initial=False):
    """Checks whether an action should be performed at a given step and frequency.

    Parameters
    ----------
    step : int
        Step number (one-indexed)
    every_n_steps : None or int
        Desired action frequency. None or 0 correspond to never executing the action.
    skip_initial : bool
        If True, frequency_check returns False at step 0.

    Returns
    -------
    decision : bool
        Whether the action should be executed.
    """

    if every_n_steps is None or every_n_steps == 0:
        return False

    if skip_initial and step == 0:
        return False

    return step % every_n_steps == 0
