"""Base classes for logits processors."""

from typing import List
import torch


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Torch method for processing logits."""
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    Class that extends list to add a specific processing method for a list of logits processors.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            scores = processor(input_ids, scores, **kwargs)
        return scores
