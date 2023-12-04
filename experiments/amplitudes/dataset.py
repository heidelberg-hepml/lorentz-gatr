import torch

class AmplitudeDataset(torch.utils.data.Dataset):

    def __init__(self, particles, amplitudes):
        self.particles = torch.tensor(particles).float()
        self.amplitudes = torch.tensor(amplitudes).float()

    def __len__(self):
        return len(self.particles)

    def __getitem__(self, idx):
        return self.particles[idx], self.amplitudes[idx]
