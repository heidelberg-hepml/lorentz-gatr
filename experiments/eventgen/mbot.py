import torch
import ot as pot

from experiments.eventgen.coordinates import StandardLogPtPhiEtaLogM2
from experiments.eventgen.helpers import ensure_angle


class MBOT(StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, virtual_components, **kwargs):
        super().__init__(**kwargs)
        self.cfm = cfm
        self.virtual_components = (
            virtual_components
            if cfm.mbot.virtual_components is None
            else cfm.mbot.virtual_components
        )

    def _get_mass(self, particle):
        # particle has to be in 'Fourmomenta' format
        unpack = lambda x: [x[..., j] for j in range(4)]
        E, px, py, pz = unpack(particle)
        mass2 = E**2 - px**2 - py**2 - pz**2

        # preprocessing
        prepd = mass2**0.5
        if self.cfm.mbot.use_logmass:
            prepd = prepd.log()

        assert torch.isfinite(prepd).all()
        return prepd

    def _get_distance(self, x1, x2):
        if self.cfm.mbot.distance == "naive":
            # Standard optimal transport (on default coordinates)
            # note that phi is cyclic -> Have to use ensure_angle
            distance = ensure_angle(x2[None, :] - x1[:, None]) ** 2
            if self.cfm.mbot.standardize:
                distance /= distance.std(dim=[0, 1])
            distance = distance.sum(dim=[-1, -2])

        elif self.cfm.mbot.distance == "mass":
            # Optimal transport on mass variables
            x1_fourmomenta = self.x_to_fourmomenta(x1)
            x2_fourmomenta = self.x_to_fourmomenta(x2)
            distance = []
            for particles in self.virtual_components:
                x1_particle = x1_fourmomenta[:, particles, :].sum(dim=-2)
                x2_particle = x2_fourmomenta[:, particles, :].sum(dim=-2)
                m1 = self._get_mass(x1_particle)
                m2 = self._get_mass(x2_particle)
                distance_single = (m2[None, :] - m1[:, None]) ** 2
                distance.append(distance_single)
            distance = torch.stack(distance, dim=-1)
            if self.cfm.mbot.standardize:
                distance /= distance.std(dim=[0, 1])
            distance = distance.sum(dim=-1)

        elif self.cfm.mbot.distance == "1d":
            # construct distance based on first entry (only for testing purposes)
            distance = (x2[None, :] - x1[:, None]) ** 2
            distance = distance[..., 0, 0]  # can play with this

        else:
            raise ValueError(
                f"Optimal transport distance {self.cfm.mbot.distance} not implemented"
            )

        if self.cfm.mbot.normalize_distance:
            distance /= distance.max()
        return distance

    def _solve_optimal_transport(self, x1, x2):
        distance = self._get_distance(x1, x2)
        distance /= distance.max()
        x1h, x2h = pot.unif(x1.shape[0], type_as=x1), pot.unif(x2.shape[0], type_as=x2)
        args = [x1h, x2h, distance]
        if self.cfm.mbot.solver == "exact":
            # exact solver (slow, because sequential)
            pi = pot.emd(*args)
        else:
            # entropy-regulated solvers (fast because parallelized; potentially unstable)
            pi = pot.sinkhorn(*args, reg=self.cfm.mbot.reg, method=self.cfm.mbot.solver)
            assert pi.std() > 1e-10, (
                f"Solver '{self.cfm.mbot.solver}' with reg={self.cfm.mbot.reg} did not converge, "
                f"consider increasing reg or using another more stable (but slower) solver "
                f"like 'sinkhorn_epsilon_scaling', 'sinkhorn_stabilized' or 'sinkhorn_log'"
            )
        p = pi.flatten()
        p /= p.sum()
        choices = torch.multinomial(p, num_samples=x1.shape[0], replacement=False)
        index1 = torch.div(choices, pi.shape[1], rounding_mode="floor")
        index2 = torch.remainder(choices, pi.shape[1])
        return index1, index2

    def get_trajectory(self, x1, x2, t):
        index1, index2 = self._solve_optimal_transport(x1, x2)
        x1, x2 = x1[index1], x2[index2]
        return super().get_trajectory(x1, x2, t)
