import pytest
import torch
from hydra import compose, initialize

import experiments.eventgen.coordinates as c
from experiments.eventgen.ttbarexperiment import ttbarExperiment
from tests.helpers import MILD_TOLERANCES as TOLERANCES


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PtPhiEtaE,
        c.PPPM,
        c.PPPM2,
        c.PPPlogM2,
        c.Jetmomenta,
        # c.Precisesiast, # TBD (weird errors)
    ],
)
def test_x_invertible(coordinates):
    """Test the the transformation from fourmomenta to x is invertible."""
    # set up experiment
    with initialize(config_path="../../../config", version_base=None):
        overrides = [
            "save=false",
            "train=false",
            "evaluate=false",
            "plot=false",
            "gatr_wrapper=GATrCFMFourmomenta",
        ]
        cfg = compose(config_name="ttbar", overrides=overrides)
        exp = ttbarExperiment(cfg)
        exp()

    # set up coordinates
    if coordinates.__name__ == "Precisesiast":
        coord = coordinates(exp.model.pt_min)
    else:
        coord = coordinates()

    # training set
    fourmomenta = exp.events_raw[0] / exp.model.units
    x = coord.fourmomenta_to_x(fourmomenta)
    fourmomenta_check = coord.x_to_fourmomenta(x)
    # print(fourmomenta-fourmomenta_check)
    torch.testing.assert_close(fourmomenta, fourmomenta_check, **TOLERANCES)

    # base distribution
    fourmomenta = (
        exp.model.sample_base(fourmomenta.shape, fourmomenta.device, fourmomenta.dtype)
    )
    x = coord.fourmomenta_to_x(fourmomenta)
    fourmomenta_check = coord.x_to_fourmomenta(x)
    # print(fourmomenta-fourmomenta_check)
    torch.testing.assert_close(fourmomenta, fourmomenta_check, **TOLERANCES)


@pytest.mark.parametrize(
    "coordinates",
    [
        c.Fourmomenta,
        c.PtPhiEtaE,
        c.PPPM,
        c.PPPM2,
        c.PPPlogM2,
        c.Jetmomenta,
        # c.Precisesiast,
    ],
)
def test_v_invertibile(coordinates):
    """Test the the transformation from fourmomenta to x is invertible."""
    # set up experiment
    with initialize(config_path="../../../config", version_base=None):
        overrides = [
            "save=false",
            "train=false",
            "evaluate=false",
            "plot=false",
            "gatr_wrapper=GATrCFMFourmomenta",
        ]
        cfg = compose(config_name="ttbar", overrides=overrides)
        exp = ttbarExperiment(cfg)
        exp()

    # set up coordinates
    if coordinates.__name__ == "Precisesiast":
        coord = coordinates(exp.model.pt_min)
    else:
        coord = coordinates()

    # training set
    fourmomenta = exp.events_raw[0] / exp.model.units
    x = coord.fourmomenta_to_x(fourmomenta)
    v_fourmomenta = torch.randn(exp.events_raw[0].shape)
    v_x = coord.velocities_fourmomenta_to_x(v_fourmomenta, fourmomenta, x)
    v_fourmomenta_check = coord.velocities_x_to_fourmomenta(v_x, x, fourmomenta)
    # print(v_fourmomenta-v_fourmomenta_check)
    torch.testing.assert_close(v_fourmomenta, v_fourmomenta_check, **TOLERANCES)

    # base distribution
    fourmomenta = (
        exp.model.sample_base(fourmomenta.shape, fourmomenta.device, fourmomenta.dtype)
    )
    x = coord.fourmomenta_to_x(fourmomenta)
    v_fourmomenta = torch.randn(exp.events_raw[0].shape)
    v_x = coord.velocities_fourmomenta_to_x(v_fourmomenta, fourmomenta, x)
    v_fourmomenta_check = coord.velocities_x_to_fourmomenta(v_x, x, fourmomenta)
    # print(v_fourmomenta-v_fourmomenta_check)
    torch.testing.assert_close(v_fourmomenta, v_fourmomenta_check, **TOLERANCES)
