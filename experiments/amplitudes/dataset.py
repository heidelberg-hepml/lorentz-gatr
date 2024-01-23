import torch


class AmplitudeDataset(torch.utils.data.Dataset):
    def __init__(self, particles, amplitudes, dtype):
        self.particles = [
            torch.tensor(particles_onedataset, dtype=dtype)
            for particles_onedataset in particles
        ]
        self.amplitudes = [
            torch.tensor(amplitudes_onedataset, dtype=dtype)
            for amplitudes_onedataset in amplitudes
        ]

        # reduce the effectively used dataset to the length of the smallest dataset
        # (pure convenience, could use more data at the cost of more code)
        self.len = min(
            [len(particles_onedataset) for particles_onedataset in self.particles]
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [
            (particles[idx], amplitudes[idx])
            for (particles, amplitudes) in zip(self.particles, self.amplitudes)
        ]
