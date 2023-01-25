# TD: [2023-01-05]: Extracted the OptimModule class from
# https://github.com/HazyResearch/state-spaces/blob/06dbbdfd0876501a7f12bf3262121badbc7658af/src/models/sequence/ss/kernel.py

import torch.nn as nn


class OptimModule(nn.Module):
    """Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters"""

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)
