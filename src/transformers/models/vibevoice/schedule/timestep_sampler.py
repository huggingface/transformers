import math
import torch


class UniformSampler:
    def __init__(self, timesteps = 1000):
        self.timesteps = timesteps
    def sample(self, batch_size, device):
        return torch.randint(0, self.timesteps, (batch_size,), device=device)
    
class LogitNormalSampler:
    def __init__(self, timesteps = 1000, m = 0, s = 1):
        self.timesteps = timesteps
        timesteps = torch.linspace(0, 1, timesteps)
        logit = torch.log(timesteps / (1 - timesteps))
        self.prob = torch.exp(-0.5 * (logit - m) ** 2 / s ** 2) / (s * math.sqrt(2 * math.pi))
    def sample(self, batch_size, device):
        return torch.multinomial(self.prob, batch_size, replacement=True).to(device)
    