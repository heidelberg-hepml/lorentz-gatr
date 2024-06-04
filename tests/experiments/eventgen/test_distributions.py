import pytest
import torch

import experiments.eventgen.coordinates as c
from experiments.eventgen.distributions import (
    StandardPPPM2,
    StandardPPPLogM2,
    FittedPPPLogM2,
    FittedLogPtPhiEtaLogM2,
)
from experiments.eventgen.processes import ttbarExperiment, zmumuExperiment
from experiments.eventgen.transforms import EPPP_to_PPPM2
from tests.helpers import TOLERANCES as TOLERANCES


@pytest.mark.parametrize(
    "distribution",
    [
        StandardPPPM2,
        StandardPPPLogM2,
        FittedPPPLogM2,
        FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [1000])
@pytest.mark.parametrize("use_delta_r_min", [False, True])
@pytest.mark.parametrize("use_pt_min", [False, True])
def test_cuts(
    distribution,
    experiment_np,
    nevents,
    use_delta_r_min,
    use_pt_min,
):
    """Test that the base distribution satisfies phase space cuts."""
    if distribution == FittedLogPtPhiEtaLogM2 and not use_pt_min:
        # this combination is not implemented
        return
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=use_delta_r_min,
        use_pt_min=use_pt_min,
    )
    d.coordinates.init_unit([nparticles])
    device = torch.device("cpu")
    dtype = torch.float32

    shape = [nevents, nparticles, 4]
    fourmomenta = d.sample(shape, device, dtype)
    mask = d.create_cut_mask(fourmomenta)
    assert mask.sum() == nevents


@pytest.mark.parametrize(
    "distribution",
    [
        StandardPPPM2,
        StandardPPPLogM2,
        FittedPPPLogM2,
        FittedLogPtPhiEtaLogM2,
    ],
)
@pytest.mark.parametrize("experiment_np", [[zmumuExperiment, 5], [ttbarExperiment, 10]])
@pytest.mark.parametrize("nevents", [1000])
@pytest.mark.parametrize("use_delta_r_min", [False, True])
@pytest.mark.parametrize("use_pt_min", [False, True])
def test_onshell(
    distribution,
    experiment_np,
    nevents,
    use_delta_r_min,
    use_pt_min,
):
    """Test that the events that should be on-shell are on-shell."""
    if distribution == FittedLogPtPhiEtaLogM2 and not use_pt_min:
        # this combination is not implemented
        return
    experiment, nparticles = experiment_np
    exp = experiment(None)
    exp.define_process_specifics()
    d = distribution(
        exp.onshell_list,
        exp.onshell_mass,
        exp.units,
        exp.delta_r_min,
        exp.pt_min,
        use_delta_r_min=use_delta_r_min,
        use_pt_min=use_pt_min,
    )
    d.coordinates.init_unit([nparticles])
    device = torch.device("cpu")
    # this test depends strongly on the dtype;
    # for torch.float64 it passes in 100% of the cases
    # for torch.float32 it fails in 10%-50%
    dtype = torch.float64

    shape = [nevents, nparticles, 4]
    fourmomenta = d.sample(shape, device, dtype)
    tr = EPPP_to_PPPM2()
    pppm2 = tr.forward(fourmomenta)
    mass = torch.sqrt(pppm2[..., 3])[..., exp.onshell_list] * exp.units
    expected_mass = (
        torch.tensor(exp.onshell_mass).unsqueeze(0).expand(mass.shape).to(dtype)
    )
    print(mass[0, ...], expected_mass[0, ...])
    torch.testing.assert_close(mass, expected_mass, **TOLERANCES)
