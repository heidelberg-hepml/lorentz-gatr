import torch
from scipy.optimize import linear_sum_assignment
from experiments.eventgen.trajectories import get_prepd_mW2, ensure_angle
import experiments.eventgen.coordinates as c


class MBOT(c.StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, **kwargs):
        super().__init__(**kwargs)
        self.cfm = cfm

    def get_trajectory(self, x1, x2, t):
        dtype = torch.float64 if self.cfm.trajs.use_float64 else torch.float32

        def get_distance(x0, x1):
            # can add more masses here
            mW0 = get_prepd_mW2(x0, c_start=self, approx_mW2=self.cfm.trajs.approx_mW2)
            mW1 = get_prepd_mW2(x1, c_start=self, approx_mW2=self.cfm.trajs.approx_mW2)
            distance = (mW0[..., None] - mW1[..., None, :]) ** 2
            return distance

        # can this be parallelized?
        x2_out = []
        subbatch_size = x1.shape[0] // self.cfm.mbot.subbatches
        subdivide = lambda x, i: x[i : i + subbatch_size]
        for i in range(self.cfm.mbot.subbatches):
            # subbatches=8 works best on my laptop
            x1_local, x2_local = subdivide(x1, i), subdivide(x2, i)
            distance = get_distance(x1_local, x2_local)
            index1, index2 = linear_sum_assignment(distance.numpy(), maximize=False)
            x2_local = x2_local[index2]
            x2_out.append(x2_local)
        x2 = torch.cat(x2_out, dim=0)
        return super().get_trajectory(x1, x2, t)
