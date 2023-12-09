# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.

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
