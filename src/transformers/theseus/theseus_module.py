import torch
from torch.distributions.bernoulli import Bernoulli

from .theseus_errors import NoSuccessorError


class TheseusModule(torch.nn.Module):
    """
    TheseusModule is the atomic replacing unit.
    """

    # TheseusModule will do nothing unless its replacing_rate is specified
    def __init__(self, predecessor: torch.nn.Module = None, successor: torch.nn.Module = None, replacing_rate=0):
        super().__init__()
        self.predecessor = predecessor
        self.successor = successor
        self.sampler = Bernoulli(torch.FloatTensor([replacing_rate]))

    def forward(self, *args, **kwargs):
        if self.successor is None:
            raise NoSuccessorError(
                "The successor is not specified. In this case, do not call `TheseusModule` directly."
            )
        return self.sample_and_pass()(*args, **kwargs)

    def sample_and_pass(self):
        # Always replace when `self.training == False`
        # Randomly substitute when `self.training == True`
        if not self.training or self.sampler.sample() == 1:
            return self.successor
        return self.predecessor

    def set_replacing_rate(self, replacing_rate):
        self.sampler = Bernoulli(torch.FloatTensor([replacing_rate]))
