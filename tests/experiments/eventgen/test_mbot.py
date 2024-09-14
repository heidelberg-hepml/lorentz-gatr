import pytest
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

from experiments.eventgen.mbot import MBOT
from experiments.eventgen.processes import ttbarExperiment
from experiments.eventgen.distributions import NaivePPPLogM2
from tests.helpers import STRICT_TOLERANCES as TOLERANCES


@pytest.mark.parametrize("batchsize", [100])
@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("solver", ["exact"])  # might not pass with 'sinkhorn'
@pytest.mark.parametrize("distance", ["naive", "1d"])
@pytest.mark.parametrize("standardize", [True, False])
def test_mbot_toy(batchsize, n, solver, distance, standardize):
    cfm_kwargs = {
        "mbot": {
            "virtual_components": None,
            "standardize": standardize,
            "distance": distance,
            "solver": solver,
            "reg": 1e-3,
            "normalize_distance": False,  # crucial for tests
        }
    }
    mbot = MBOT(
        cfm=OmegaConf.create(cfm_kwargs),
        virtual_components=None,
        pt_min=[0],
        units=1,
        onshell_list=[0],
    )

    shape = [2, 1]
    for _ in range(n):
        x1, x2 = torch.randn(batchsize, *shape), torch.randn(batchsize, *shape)

        index1, index2 = mbot._solve_optimal_transport(x1, x2)
        arange = torch.arange(0, len(index1))

        distance_raw = mbot._get_distance(x1, x2)
        index1_check, index2_check = linear_sum_assignment(
            distance_raw.numpy(), maximize=False
        )
        distance_naive = mbot._get_distance(x1, x2)[arange, arange].sum()
        distance_ours = mbot._get_distance(x1, x2)[index1, index2].sum()
        distance_check = mbot._get_distance(x1, x2)[index1_check, index2_check].sum()

        if solver == "exact":
            # check if exact pot solver agrees with scipy implementation
            torch.testing.assert_close(distance_ours, distance_check, **TOLERANCES)

        # check that distance decreased
        assert distance_ours <= distance_naive


@pytest.mark.parametrize("batchsize", [100])
@pytest.mark.parametrize("n", [10])
@pytest.mark.parametrize("solver", ["exact"])  # might not pass with 'sinkhorn'
@pytest.mark.parametrize("distance", ["naive", "1d", "mass"])
@pytest.mark.parametrize("standardize", [True, False])
def test_mbot_realistic(batchsize, n, solver, distance, standardize):
    exp, nparticles = ttbarExperiment(None), 6
    exp.define_process_specifics()
    cfm_kwargs = {
        "mbot": {
            "virtual_components": exp.virtual_components,
            "standardize": standardize,
            "distance": distance,
            "solver": solver,
            "reg": 1e-3,
            "normalize_distance": False,  # important for tests
            "use_logmass": True,
        }
    }
    mbot = MBOT(
        cfm=OmegaConf.create(cfm_kwargs),
        virtual_components=None,
        pt_min=exp.pt_min,
        units=exp.units,
        onshell_list=exp.onshell_list,
    )
    d = NaivePPPLogM2(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    d.coordinates.init_unit([nparticles])
    shape = (batchsize, nparticles, 4)
    device, dtype = torch.device("cpu"), torch.float32
    sample_kwargs = {
        "shape": shape,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    x_init = d.sample(**sample_kwargs)
    mbot.init_fit([x_init])
    for _ in range(n):
        x1, x2 = d.sample(**sample_kwargs), d.sample(**sample_kwargs)

        index1, index2 = mbot._solve_optimal_transport(x1, x2)
        arange = torch.arange(0, len(index1))

        distance_raw = mbot._get_distance(x1, x2)
        index1_check, index2_check = linear_sum_assignment(
            distance_raw.numpy(), maximize=False
        )
        distance_naive = mbot._get_distance(x1, x2)[arange, arange].sum()
        distance_ours = mbot._get_distance(x1, x2)[index1, index2].sum()
        distance_check = mbot._get_distance(x1, x2)[index1_check, index2_check].sum()

        if solver == "exact":
            torch.testing.assert_close(distance_ours, distance_check, **TOLERANCES)

        assert distance_ours < distance_naive
