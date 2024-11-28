import torch

from gatr.interface import embed_vector, extract_vector
from tests_regression.regression_datasets.constants import DATASET_SIZE, DEVICE


class TopReconstructionDataset(torch.utils.data.Dataset):
    """Toy dataset for reconstruction the mass of a top quark and W boson from
    the 4-momenta of 3 arbitrarily ordered quarks. The decay chain is t > W b, W > q q'.
    """

    def __init__(
        self,
        mW_mean=80.4,
        mW_std=5.0,
        mt_mean=173.0,
        mt_std=20.0,
        mq_std=10.0,
        p_std=10.0,
        safety_factor=10,
    ):
        """
        This is ugly...
        Very happy about suggestions for how to do it nicer.
        """
        super().__init__()
        DATASET_SIZE_TEST = DATASET_SIZE * safety_factor

        while True:
            # sample masses
            mt = torch.randn(DATASET_SIZE_TEST) * mt_std + mt_mean
            mW = torch.randn(DATASET_SIZE_TEST) * mW_std + mW_mean
            mq = torch.randn(3, DATASET_SIZE_TEST) * mq_std + 0.0
            mq[mq <= 0] = -mq[mq <= 0]
            mb, mq1, mq2 = mq

            # sample top 4-momentum
            pt = torch.randn(DATASET_SIZE_TEST, 4) * p_std
            iflat = torch.randint(low=1, high=4, size=(DATASET_SIZE_TEST,))
            inotflat = torch.stack((1 + (iflat) % 3, 1 + (iflat + 1) % 3), dim=-1)
            ihelper = torch.arange(DATASET_SIZE_TEST)
            pt[ihelper, iflat] = 0.0
            pt[:, 0] = torch.sqrt(mt**2 + torch.sum(pt[:, 1:] ** 2, dim=-1))

            # sample W and b with the constraint pW+pb=pt
            pW = torch.randn(DATASET_SIZE_TEST, 4) * p_std
            iflat2 = 1 + (iflat + 1) % 3
            inotflat2 = torch.stack((1 + (iflat2) % 3, 1 + (iflat2 + 1) % 3), dim=-1)
            pW[ihelper, iflat2] = 0.0
            pW[:, 0] = (
                1
                / (2 * pt[:, 0])
                * (
                    mW**2
                    - mb**2
                    + pt[:, 0] ** 2
                    + torch.sum(
                        pW[ihelper[:, None], inotflat] ** 2
                        - (
                            pt[ihelper[:, None], inotflat]
                            - pW[ihelper[:, None], inotflat]
                        )
                        ** 2,
                        dim=-1,
                    )
                )
            )
            pW[ihelper, iflat] = torch.sqrt(
                pW[:, 0] ** 2
                - mW**2
                - torch.sum(pW[ihelper[:, None], inotflat] ** 2, dim=-1)
            )
            pb = pt - pW

            # sample q1 and q2 with the constraint pq1+pq2=pW
            pq1 = torch.randn(DATASET_SIZE_TEST, 4) * p_std
            pq1[:, 0] = (
                1
                / (2 * pW[:, 0])
                * (
                    mq1**2
                    - mq2**2
                    + pW[:, 0] ** 2
                    + torch.sum(
                        pq1[ihelper[:, None], inotflat2] ** 2
                        - (
                            pW[ihelper[:, None], inotflat2]
                            - pq1[ihelper[:, None], inotflat2]
                        )
                        ** 2,
                        dim=-1,
                    )
                )
            )
            pq1[ihelper, iflat2] = torch.sqrt(
                pq1[:, 0] ** 2
                - mq1**2
                - torch.sum(pq1[ihelper[:, None], inotflat2] ** 2, dim=-1)
            )
            pq2 = pW - pq1

            # collect valid events
            mask = torch.all(torch.isfinite(pW) * torch.isfinite(pq1), dim=-1)
            pt, pW, pb, pq1, pq2 = (
                pt[mask, :],
                pW[mask, :],
                pb[mask, :],
                pq1[mask, :],
                pq2[mask, :],
            )
            mt, mW, mb, mq1, mq2 = mt[mask], mW[mask], mb[mask], mq1[mask], mq2[mask]

            print(
                f"Obtained {mask.sum(dim=0)}/{DATASET_SIZE} valid events using safety_factor={safety_factor}"
            )
            if mask.sum(dim=0) > DATASET_SIZE:
                break

        # consistency checks
        def check_momentum_conservation(p1, p2, p3):
            diff = p1 - (p2 + p3)
            mask = (diff > 0.1).any(dim=-1)
            assert (torch.abs(diff) < 0.1).all()

        check_momentum_conservation(pt, pW, pb)
        check_momentum_conservation(pW, pq1, pq2)

        def check_onshell(p, m):
            p_squared = p[:, 0] ** 2 - torch.sum(p[:, 1:] ** 2, dim=-1)
            diff = p_squared - m**2
            assert (torch.abs(diff) < 0.1).all()

        for p, m in [(pt, mt), (pW, mW), (pb, mb), (pq1, mq1), (pq2, mq2)]:
            check_onshell(p, m)

        # collect results
        self.event = torch.stack((pb, pq1, pq2), dim=1)
        self.reco = torch.stack((pt, pW), dim=1)

        # shuffle quarks in the event
        idx = torch.randperm(self.event.size(1))
        self.event = self.event[:, idx]

        # reduce dataset
        self.reco = self.reco[:DATASET_SIZE, ...]
        self.event = self.event[:DATASET_SIZE, ...]

        # normalize (= change units)
        scale = self.event.mean()
        self.reco /= scale
        self.event /= scale

        # If there's space on the GPU, let's keep the data on the GPU
        try:
            self.reco.to(DEVICE)
            self.event.to(DEVICE)
        except RuntimeError:
            pass

    def __len__(self):
        """Return number of samples."""
        return len(self.event)

    def __getitem__(self, idx):
        """Return datapoint."""
        return self.event[idx], self.reco[idx]


class TopReconstructionWrapper(torch.nn.Module):
    """Wrapper around GATr networks for TopReconstructionDataset."""

    mv_in_channels = 1
    mv_out_channels = 2
    s_in_channels = 1
    s_out_channels = 1

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs: torch.Tensor):
        """Wrapped forward pass pass.

        Parses inputs into GA + scalar representation, calls the forward pass of the wrapped net,
        and extracts the outputs from the GA + scalar representation again.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        """

        multivector, scalars = self.embed_into_ga(inputs)
        multivector_outputs, scalar_outputs = self.net(
            multivector,
            scalars=scalars,
        )
        outputs = self.extract_from_ga(multivector_outputs, scalar_outputs)

        return outputs

    def embed_into_ga(self, inputs):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw inputs, as given by dataset.

        Returns
        -------
        mv_inputs : torch.Tensor
            Multivector inputs, as expected by geometric network.
        scalar_inputs : torch.Tensor or None
            Scalar inputs, as expected by geometric network.
        """
        batchsize, num_objects, num_features = inputs.shape
        assert num_objects == 3
        assert num_features == 4

        multivector = embed_vector(inputs)  # (batchsize, 3, 16)
        multivector = multivector.unsqueeze(2)  # (batchsize, 3, 1, 16)

        scalars = torch.zeros(
            (batchsize, 3, 1), device=inputs.device
        )  # (batchsize, 3, 1)
        return multivector, scalars

    def extract_from_ga(self, multivector, scalars):
        """Embeds raw inputs into the geometric algebra (+ scalar) representation.

        To be implemented by subclasses.

        Parameters
        ----------
        multivector : torch.Tensor
            Multivector outputs from geometric network.
        scalars : torch.Tensor or None
            Scalar outputs from geometric network.

        Returns
        -------
        outputs : torch.Tensor
            Raw outputs, as expected in dataset.
        """

        _, num_objects, num_channels, num_ga_components = multivector.shape
        assert num_objects == 3
        assert num_channels == 2
        assert num_ga_components == 16

        pt = extract_vector(multivector[:, :, 0, :])  # (batchsize, 3, 4)
        pW = extract_vector(multivector[:, :, 1, :])  # (batchsize, 3, 4)
        reco = torch.stack((pt, pW), dim=1)
        reco = reco[:, :, 0, :]  # pick first output channel
        # reco = reco.mean(dim=2) # average over output channels (much worse, probably because mean breaks symmetry)
        return reco
