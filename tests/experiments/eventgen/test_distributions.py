import pytest
import torch

import experiments.eventgen.coordinates as c
from experiments.eventgen.distributions import (
    FourmomentaDistribution,
    JetmomentaDistribution,
    NaiveDistribution,
)
from experiments.eventgen.helpers import get_mass
from tests.helpers import MILD_TOLERANCES as TOLERANCES

ttbar_base_kwargs = {
    "pxy_std": 61.14,
    "pz_std": 286.26,
    "logpt_mean": 4.06,
    "logpt_std": 0.59,
    "logmass_mean": 2.15,
    "logmass_std": 0.71,
    "eta_std": 1.51,
}


@pytest.mark.parametrize(
    "distribution",
    [
        FourmomentaDistribution,
        JetmomentaDistribution,
        NaiveDistribution,
    ],
)
@pytest.mark.parametrize("nevents", [100000])
@pytest.mark.parametrize("onshell_list", [[], [0, 1]])
@pytest.mark.parametrize("onshell_mass", [[], [10.0, 5.0]])
@pytest.mark.parametrize("units", [206.6])
@pytest.mark.parametrize("delta_r_min", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("pt_min", [[20.0] * 6])
@pytest.mark.parametrize("nparticles", [6])
def test_cuts(
    distribution,
    nevents,
    onshell_list,
    onshell_mass,
    units,
    delta_r_min,
    pt_min,
    nparticles,
):
    """Test that the base distribution satisfies phase space cuts."""
    if len(onshell_list) != len(onshell_mass):
        # only do meaningful tests
        return
    d = distribution(
        onshell_list,
        onshell_mass,
        units,
        ttbar_base_kwargs,
        delta_r_min,
        pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    device = torch.device("cpu")
    dtype = torch.float32

    shape = [nevents, nparticles, 4]
    fourmomenta = d.sample(shape, device, dtype)
    mask = d.create_cut_mask(fourmomenta)
    assert mask.sum() == nevents


@pytest.mark.parametrize(
    "distribution",
    [
        FourmomentaDistribution,
        JetmomentaDistribution,
    ],
)
@pytest.mark.parametrize("nevents", [100000])
@pytest.mark.parametrize("onshell_list", [[0, 1]])
@pytest.mark.parametrize("onshell_mass", [[10.0, 5.0]])
@pytest.mark.parametrize("units", [206.6])
@pytest.mark.parametrize("delta_r_min", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("pt_min", [[20.0] * 6])
@pytest.mark.parametrize("nparticles", [6])
def test_onshell(
    distribution,
    nevents,
    onshell_list,
    onshell_mass,
    units,
    delta_r_min,
    pt_min,
    nparticles,
):
    """Test that the events that should be on-shell are on-shell."""
    if len(onshell_list) != len(onshell_mass):
        # only do meaningful tests
        return
    d = distribution(
        onshell_list,
        onshell_mass,
        units,
        ttbar_base_kwargs,
        delta_r_min,
        pt_min,
        use_delta_r_min=True,
        use_pt_min=True,
    )
    device = torch.device("cpu")
    dtype = torch.float32

    shape = [nevents, nparticles, 4]
    fourmomenta = d.sample(shape, device, dtype) * units
    mass = get_mass(fourmomenta)[..., onshell_list]
    expected_mass = torch.tensor(onshell_mass).unsqueeze(0).expand(mass.shape)
    print(mass, expected_mass)
    torch.testing.assert_close(mass, expected_mass, **TOLERANCES)
