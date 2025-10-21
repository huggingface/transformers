from typing import List, Optional
import torch

from ..moderation.base import SafetyChecker, SafetyConfig, SafetyResult
from .logits_process import LogitsProcessor
from .stopping_criteria import StoppingCriteria


class ModerationLogitsProcessor(LogitsProcessor):
    """A LogitsProcessor that queries a SafetyChecker on decoded text and suppresses forbidden token ids.

    This processor requires the caller to provide the tokenizer and the partial decoded text for each batch
    so that the safety checker can inspect the current candidate sequences. In many generation loops this is
    available as `input_ids` and a tokenizer `batch_decode`.
    """

    def __init__(self, safety_checker: SafetyChecker, tokenizer, safety_config: Optional[SafetyConfig] = None):
        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.safety_config = safety_config or SafetyConfig()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch, seq_len)
        # scores: (batch, vocab)
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        results = self.safety_checker.check_texts(texts)

        for i, res in enumerate(results):
            if not res.is_safe:
                for tid in res.forbidden_token_ids:
                    if 0 <= tid < scores.size(-1):
                        scores[i, tid] = -float("inf")

        return scores


class ModerationStoppingCriteria(StoppingCriteria):
    """Stops generation when safety score falls below the configured threshold."""

    def __init__(self, safety_checker: SafetyChecker, tokenizer, safety_config: Optional[SafetyConfig] = None):
        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.safety_config = safety_config or SafetyConfig()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # We inspect the decoded text and consult the safety checker. If any text is unsafe and
        # stop_on_unsafe is True, return True to stop.
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        results = self.safety_checker.check_texts(texts)
        for res in results:
            if not res.is_safe and self.safety_config.stop_on_unsafe and res.score < self.safety_config.threshold:
                return True
        return False


__all__ = ["ModerationLogitsProcessor", "ModerationStoppingCriteria"]
