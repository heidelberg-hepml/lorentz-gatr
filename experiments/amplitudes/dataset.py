import torch

class AmplitudeDataset(torch.utils.data.Dataset):

    def __init__(self, particles, amplitudes, dtype):
        self.particles = torch.tensor(particles, dtype=dtype)
        self.amplitudes = torch.tensor(amplitudes, dtype=dtype)

    def __len__(self):
        return self.particles.shape[0]

    def __getitem__(self, idx):
        return self.particles[idx], self.amplitudes[idx]
