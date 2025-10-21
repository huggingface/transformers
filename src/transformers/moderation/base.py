from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List


@dataclass
class SafetyConfig:
    """Configuration for moderation behavior.

    Attributes:
        threshold: float - score below which text is considered unsafe (0..1)
        stop_on_unsafe: bool - whether generation should be stopped when unsafe content is detected
    """
    threshold: float = 0.5
    stop_on_unsafe: bool = True


@dataclass
class SafetyResult:
    """Result returned by a SafetyChecker for a single text."""
    is_safe: bool
    score: float
    # Token ids that should be suppressed (e.g., bad words detected).
    forbidden_token_ids: List[int]


class SafetyChecker(ABC):
    """Abstract base class for safety checkers.

    Implementations should provide a fast `check_texts` method that accepts an iterable of
    strings and returns a list of `SafetyResult`, one per input.
    """

    @abstractmethod
    def check_texts(self, texts: List[str]) -> List[SafetyResult]:
        """Check a batch of texts and return SafetyResult per text."""
        raise NotImplementedError()
