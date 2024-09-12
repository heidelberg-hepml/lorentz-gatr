import torch
import ot as pot

from experiments.eventgen.trajectories import get_prepd_mW2
from experiments.eventgen.coordinates import StandardLogPtPhiEtaLogM2


class MBOT(StandardLogPtPhiEtaLogM2):
    def __init__(self, cfm, **kwargs):
        super().__init__(**kwargs)
        self.cfm = cfm

    def get_trajectory(self, x1, x2, t):
        dtype = torch.float64 if self.cfm.trajs.use_float64 else torch.float32

        def get_distance(x0, x1):
            # can add more masses here
            mW0 = get_prepd_mW2(
                x0,
                c_start=self,
                approx_mW2=self.cfm.trajs.approx_mW2,
                use_logmW2=self.cfm.trajs.use_logmW2,
            )
            mW1 = get_prepd_mW2(
                x1,
                c_start=self,
                approx_mW2=self.cfm.trajs.approx_mW2,
                use_logmW2=self.cfm.trajs.use_logmW2,
            )
            distance = (mW0[..., None] - mW1[..., None, :]) ** 2
            return distance

        x2_out = []
        minibatch_size = x1.shape[0] // self.cfm.mbot.minibatches
        subdivide = lambda x, i: x[i : i + minibatch_size]
        for i in range(self.cfm.mbot.minibatches):
            x1_local, x2_local = subdivide(x1, i), subdivide(x2, i)
            distance = get_distance(x1_local, x2_local)
            distance /= distance.max()
            x1h_local, x2h_local = pot.unif(
                x1_local.shape[0], type_as=x1_local
            ), pot.unif(x2_local.shape[0], type_as=x2_local)
            if float(self.cfm.mbot.reg) <= 0.0:
                # use reg=-1 to handle the emd/sinkhorn? and reg? questions with one parameter
                # using reg<=0 in pot.sinkhorn does not work
                pi = pot.emd(x1h_local, x2h_local, distance)
            else:
                pi = pot.sinkhorn(x1h_local, x2h_local, distance, reg=self.cfm.mbot.reg)
            p = pi.flatten()
            p /= p.sum()
            p = p.cpu()
            choices = torch.multinomial(
                p, num_samples=x1_local.shape[0], replacement=False
            )
            index2 = torch.remainder(choices, pi.shape[1])
            index2 = torch.tensor(index2, device=x1.device, dtype=torch.long)
            x2_local = x2_local[index2]
            x2_out.append(x2_local)
        x2 = torch.cat(x2_out, dim=0)
        return super().get_trajectory(x1, x2, t)
