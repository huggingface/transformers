import torch
from torch import nn


class RevIN(nn.Module):
    def __init__(self, start_dim=1, eps=1e-5, denorm_channels: list = None):
        """
        :param start_dim: it is 1 if [bs x seq_len x nvars], it is 3 is [bs x tsg1 x tsg2 x seq_len x n_vars]
        :denorm_channels if the denorm input shape has less number of channels, mention the channels in the denorm
        input here.
        """
        super(RevIN, self).__init__()
        self.start_dim = start_dim
        self.denorm_channels = denorm_channels
        self.eps = eps

    def set_statistics(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        elif mode == "transform":
            x = self._normalize(x)

        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(self.start_dim, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):

        if self.denorm_channels is None:
            x = x * self.stdev
            x = x + self.mean
        else:
            x = x * self.stdev[..., self.denorm_channels]
            x = x + self.mean[..., self.denorm_channels]

        return x
