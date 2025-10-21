# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Safety checking infrastructure for text generation.

This module provides pluggable safety checkers that can be used to moderate
both inputs and outputs during text generation, similar to safety features
in production LLMs.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from ..utils import add_start_docstrings, logging
from .logits_process import LogitsProcessor
from .stopping_criteria import StoppingCriteria


logger = logging.get_logger(__name__)


# Mapping of string names to safety checker classes
SAFETY_CHECKER_MAPPING = {}  # Will be populated after class definitions


def get_safety_checker(
    checker: Union[str, "SafetyChecker", None],
    checker_kwargs: Optional[Dict[str, Any]] = None
) -> Optional["SafetyChecker"]:
    """
    Get safety checker from string name or return instance.

    Args:
        checker (`Union[str, SafetyChecker, None]`):
            Safety checker name, instance, or None.
        checker_kwargs (`Dict[str, Any]`, *optional*):
            Kwargs for instantiating the checker if `checker` is a string.

    Returns:
        `Optional[SafetyChecker]`: SafetyChecker instance or None.

    Raises:
        ValueError: If checker string is not recognized.
        TypeError: If checker is not a valid type.

    Example:
        ```python
        # From string
        checker = get_safety_checker("keyword", {"blocked_keywords": ["unsafe"]})

        # From instance
        checker = get_safety_checker(KeywordSafetyChecker(blocked_keywords=["unsafe"]))

        # None
        checker = get_safety_checker(None)  # Returns None
        ```
    """
    if checker is None:
        return None

    if isinstance(checker, SafetyChecker):
        return checker

    if isinstance(checker, str):
        if checker not in SAFETY_CHECKER_MAPPING:
            raise ValueError(
                f"Unknown safety checker: '{checker}'. "
                f"Available checkers: {list(SAFETY_CHECKER_MAPPING.keys())}"
            )
        checker_class = SAFETY_CHECKER_MAPPING[checker]
        return checker_class(**(checker_kwargs or {}))

    raise TypeError(
        f"safety_checker must be str, SafetyChecker instance, or None, got {type(checker)}"
    )


@dataclass
class SafetyCheckResult:
    """
    Result of a safety check operation.

    Args:
        is_safe (`bool`):
            Whether the content passed the safety check.
        violation_categories (`List[str]`, *optional*, defaults to `[]`):
            List of safety categories that were violated.
        confidence_scores (`Dict[str, float]`, *optional*, defaults to `{}`):
            Confidence scores for each category checked.
        filtered_text (`str`, *optional*):
            If filtering is enabled, the text after filtering unsafe content.
        metadata (`Dict[str, Any]`, *optional*, defaults to `{}`):
            Additional metadata about the safety check.
    """

    is_safe: bool
    violation_categories: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    filtered_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyChecker(ABC):
    """
    Abstract base class for safety checkers.

    Safety checkers can be used to moderate both inputs and outputs during text generation.
    Subclass this to implement custom safety checking logic.

    Example:
        ```python
        class CustomSafetyChecker(SafetyChecker):
            def check_input(self, text, **kwargs):
                # Your input validation logic
                is_safe = your_validation_function(text)
                return SafetyCheckResult(is_safe=is_safe)

            def check_output(self, text, context=None, **kwargs):
                # Your output validation logic
                is_safe = your_validation_function(text, context)
                return SafetyCheckResult(is_safe=is_safe)
        ```
    """

    @abstractmethod
    def check_input(self, text: Union[str, List[str]], **kwargs) -> Union[SafetyCheckResult, List[SafetyCheckResult]]:
        """
        Check if input text is safe.

        Args:
            text (`Union[str, List[str]]`):
                The input text(s) to check.
            **kwargs:
                Additional arguments specific to the safety checker.

        Returns:
            `Union[SafetyCheckResult, List[SafetyCheckResult]]`:
                Safety check result(s).
        """
        raise NotImplementedError("Subclasses must implement check_input")

    @abstractmethod
    def check_output(
        self,
        text: Union[str, List[str]],
        context: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Union[SafetyCheckResult, List[SafetyCheckResult]]:
        """
        Check if output text is safe given the context.

        Args:
            text (`Union[str, List[str]]`):
                The generated text(s) to check.
            context (`Union[str, List[str]]`, *optional*):
                The input context(s) that led to this output.
            **kwargs:
                Additional arguments specific to the safety checker.

        Returns:
            `Union[SafetyCheckResult, List[SafetyCheckResult]]`:
                Safety check result(s).
        """
        raise NotImplementedError("Subclasses must implement check_output")


class KeywordSafetyChecker(SafetyChecker):
    """
    Simple safety checker based on keyword matching.

    This checker blocks content containing specific keywords or patterns.
    It's a basic implementation suitable for simple use cases.

    Args:
        blocked_keywords (`List[str]`):
            List of keywords to block. Matching is case-insensitive.
        blocked_patterns (`List[str]`, *optional*):
            List of regex patterns to block.
        category_name (`str`, *optional*, defaults to `"blocked_content"`):
            Name of the violation category for reporting.

    Example:
        ```python
        safety_checker = KeywordSafetyChecker(
            blocked_keywords=["violence", "explicit"],
            blocked_patterns=[r"\\b(\\w+)\\1{2,}\\b"]  # Repeated words
        )
        result = safety_checker.check_output("This is a test")
        print(result.is_safe)  # True
        ```
    """

    def __init__(
        self,
        blocked_keywords: List[str],
        blocked_patterns: Optional[List[str]] = None,
        category_name: str = "blocked_content",
    ):
        self.blocked_keywords = set(kw.lower() for kw in blocked_keywords)
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (blocked_patterns or [])]
        self.category_name = category_name

    def _check_text(self, text: str) -> SafetyCheckResult:
        """Check a single text string for safety violations."""
        text_lower = text.lower()

        # Check keywords
        for keyword in self.blocked_keywords:
            if keyword in text_lower:
                return SafetyCheckResult(
                    is_safe=False,
                    violation_categories=[self.category_name],
                    metadata={"matched_keyword": keyword},
                )

        # Check patterns
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return SafetyCheckResult(
                    is_safe=False,
                    violation_categories=[self.category_name],
                    metadata={"matched_pattern": pattern.pattern},
                )

        return SafetyCheckResult(is_safe=True)

    def check_input(self, text: Union[str, List[str]], **kwargs) -> Union[SafetyCheckResult, List[SafetyCheckResult]]:
        if isinstance(text, str):
            return self._check_text(text)
        return [self._check_text(t) for t in text]

    def check_output(
        self,
        text: Union[str, List[str]],
        context: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Union[SafetyCheckResult, List[SafetyCheckResult]]:
        # For keyword checker, context doesn't matter
        return self.check_input(text, **kwargs)


class SafetyStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that halts generation when safety violations are detected.

    This stopping criteria uses a `SafetyChecker` to determine if the generated
    content violates safety guidelines and stops generation accordingly.

    Args:
        safety_checker (`SafetyChecker`):
            The safety checker to use for validation.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to decode sequences for safety checking.
        safety_config (`SafetyConfig`):
            Configuration for safety checking.
        prompt_length (`int`, *optional*):
            Length of the prompt to exclude from safety checking.

    Example:
        ```python
        from transformers import AutoTokenizer
        from transformers.generation import KeywordSafetyChecker, SafetyConfig, SafetyStoppingCriteria

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        safety_checker = KeywordSafetyChecker(blocked_keywords=["unsafe"])
        safety_config = SafetyConfig(safety_checker=safety_checker)

        stopping_criteria = SafetyStoppingCriteria(
            safety_checker=safety_checker,
            tokenizer=tokenizer,
            safety_config=safety_config
        )
        ```
    """

    def __init__(
        self,
        safety_checker: SafetyChecker,
        tokenizer,
        generation_config: "GenerationConfig",
        prompt_length: Optional[int] = None,
    ):
        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.prompt_length = prompt_length or 0
        self.check_counter = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        Check if generation should stop due to safety violations.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Generated sequences so far.
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores (not used in safety checking).
            **kwargs:
                Additional arguments.

        Returns:
            `torch.BoolTensor` of shape `(batch_size,)`:
                `True` for sequences that should stop due to safety violations.
        """
        # Only check at specified frequency
        self.check_counter += 1
        if self.check_counter % self.generation_config.safety_check_frequency != 0:
            return torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)

        batch_size = input_ids.shape[0]
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # Decode sequences (excluding prompt if specified)
        start_idx = self.prompt_length if self.prompt_length > 0 else 0
        texts = self.tokenizer.batch_decode(input_ids[:, start_idx:], skip_special_tokens=True)

        # Check each sequence
        results = self.safety_checker.check_output(texts)
        if not isinstance(results, list):
            results = [results]

        for idx, result in enumerate(results):
            if not result.is_safe:
                should_stop[idx] = True
                logger.warning(
                    f"Safety violation detected in sequence {idx}. "
                    f"Categories: {result.violation_categories}. Generation stopped."
                )

        return should_stop


SAFETY_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head.

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
"""


class SafetyLogitsProcessor(LogitsProcessor):
    """
    Logits processor that reduces probabilities of tokens leading to unsafe content.

    This processor attempts to prevent generation of unsafe content by reducing
    the logits/probabilities of tokens that would create safety violations.

    Args:
        safety_checker (`SafetyChecker`):
            The safety checker to use for validation.
        tokenizer (`PreTrainedTokenizerBase`):
            Tokenizer to decode sequences and candidate tokens.
        safety_config (`SafetyConfig`):
            Configuration for safety checking.
        penalty_value (`float`, *optional*, defaults to `-inf`):
            Value to set for unsafe token logits. Use `-inf` to completely block,
            or a large negative number (e.g., -100) for strong discouragement.

    Example:
        ```python
        from transformers import AutoTokenizer
        from transformers.generation import KeywordSafetyChecker, SafetyConfig, SafetyLogitsProcessor

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        safety_checker = KeywordSafetyChecker(blocked_keywords=["unsafe"])
        safety_config = SafetyConfig(safety_checker=safety_checker)

        processor = SafetyLogitsProcessor(
            safety_checker=safety_checker,
            tokenizer=tokenizer,
            safety_config=safety_config
        )
        ```

    Note:
        This processor can be computationally expensive as it needs to check
        multiple potential continuations. Consider using it with
        `safety_config.check_frequency > 1` for better performance.
    """

    def __init__(
        self,
        safety_checker: SafetyChecker,
        tokenizer,
        generation_config: "GenerationConfig",
        penalty_value: Optional[float] = None,
    ):
        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.penalty_value = penalty_value if penalty_value is not None else generation_config.safety_penalty_value
        self.check_counter = 0

    @add_start_docstrings(SAFETY_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Only check at specified frequency
        self.check_counter += 1
        if self.check_counter % self.generation_config.safety_check_frequency != 0:
            return scores

        # For performance, we use a simpler heuristic:
        # Check if the current sequence + most likely next tokens would be safe
        # This is a trade-off between safety and performance

        batch_size, vocab_size = scores.shape

        # Get top-k most likely tokens to check
        top_k = min(50, vocab_size)  # Check top 50 tokens
        top_scores, top_indices = torch.topk(scores, top_k, dim=-1)

        for batch_idx in range(batch_size):
            current_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=True)

            unsafe_token_indices = []

            # Check each top candidate token
            for token_idx in top_indices[batch_idx]:
                token_text = self.tokenizer.decode(token_idx, skip_special_tokens=True)
                candidate_text = current_text + token_text

                # Quick safety check
                result = self.safety_checker.check_output(candidate_text)
                if not result.is_safe:
                    unsafe_token_indices.append(token_idx.item())

            # Penalize unsafe tokens
            if unsafe_token_indices:
                scores[batch_idx, unsafe_token_indices] = self.penalty_value

        return scores


# Populate the safety checker mapping
SAFETY_CHECKER_MAPPING["keyword"] = KeywordSafetyChecker
