# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


@dataclass
class SafetyViolation:
    """
    Represents a single safety violation detected in text.

    Args:
        category (`str`):
            The category of safety violation (e.g., "toxicity", "bias", "pii").
        confidence (`float`):
            Confidence score for the violation detection, ranging from 0.0 to 1.0.
        severity (`str`, *optional*, defaults to `"medium"`):
            Severity level of the violation. One of "low", "medium", "high", "critical".
        description (`str`, *optional*, defaults to `""`):
            Human-readable description of the violation.
        span (`Tuple[int, int]`, *optional*):
            Character span in the original text where the violation occurs, if applicable.
    """

    category: str
    confidence: float
    severity: str = "medium"
    description: str = ""
    span: Optional[tuple[int, int]] = None


@dataclass
class SafetyResult:
    """
    Result of a safety checking operation.

    Args:
        is_safe (`bool`):
            Whether the checked text is considered safe overall.
        confidence (`float`):
            Overall confidence in the safety assessment, ranging from 0.0 to 1.0.
        violations (`List[SafetyViolation]`):
            List of safety violations detected in the text.
        metadata (`Dict[str, Any]`):
            Additional checker-specific information and context.
    """

    is_safe: bool
    confidence: float
    violations: list[SafetyViolation]
    metadata: dict[str, Any]


@dataclass
class SafetyMetrics:
    """
    Metrics collection for safety operations monitoring and analysis.

    Tracks performance and usage statistics for safety checking operations,
    enabling production monitoring and optimization.

    Args:
        total_generations (`int`, defaults to 0):
            Total number of generations attempted.
        blocked_generations (`int`, defaults to 0):
            Number of generations blocked due to safety violations.
        suppression_events (`int`, defaults to 0):
            Number of token suppression events during generation.
        cache_hits (`int`, defaults to 0):
            Number of cache hits for safety check results.
        cache_misses (`int`, defaults to 0):
            Number of cache misses requiring new safety checks.
        total_safety_check_time_ms (`float`, defaults to 0.0):
            Cumulative time spent on safety checks in milliseconds.
        safety_check_count (`int`, defaults to 0):
            Total number of safety checks performed.
    """

    total_generations: int = 0
    blocked_generations: int = 0
    suppression_events: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_safety_check_time_ms: float = 0.0
    safety_check_count: int = 0

    def __post_init__(self):
        """Initialize thread safety lock after dataclass fields."""
        self._lock = threading.Lock()

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total_cache_ops = self.cache_hits + self.cache_misses
        if total_cache_ops == 0:
            return 0.0
        return (self.cache_hits / total_cache_ops) * 100.0

    @property
    def avg_safety_check_time_ms(self) -> float:
        """Calculate average safety check time in milliseconds."""
        if self.safety_check_count == 0:
            return 0.0
        return self.total_safety_check_time_ms / self.safety_check_count

    @property
    def block_rate(self) -> float:
        """Calculate generation block rate as a percentage."""
        if self.total_generations == 0:
            return 0.0
        return (self.blocked_generations / self.total_generations) * 100.0

    def record_safety_check(self, check_time_ms: float) -> None:
        """Record a safety check operation with timing."""
        with self._lock:
            self.safety_check_count += 1
            self.total_safety_check_time_ms += check_time_ms

    def record_cache_hit(self) -> None:
        """Record a cache hit event."""
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss event."""
        with self._lock:
            self.cache_misses += 1

    def record_generation_attempt(self) -> None:
        """Record a generation attempt."""
        with self._lock:
            self.total_generations += 1

    def record_blocked_generation(self) -> None:
        """Record a generation that was blocked due to safety violations."""
        with self._lock:
            self.blocked_generations += 1

    def record_suppression_event(self) -> None:
        """Record a token suppression event."""
        with self._lock:
            self.suppression_events += 1

    def to_dict(self) -> dict[str, Union[int, float]]:
        """
        Export metrics as dictionary for logging or monitoring systems.

        Returns:
            Dict[str, Union[int, float]]: Dictionary containing all metrics.
        """
        with self._lock:
            return {
                "total_generations": self.total_generations,
                "blocked_generations": self.blocked_generations,
                "suppression_events": self.suppression_events,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hit_rate,
                "avg_safety_check_time_ms": self.avg_safety_check_time_ms,
                "block_rate": self.block_rate,
                "safety_check_count": self.safety_check_count,
            }

    def reset(self) -> None:
        """Reset all metrics to zero for new measurement period."""
        with self._lock:
            self.total_generations = 0
            self.blocked_generations = 0
            self.suppression_events = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_safety_check_time_ms = 0.0
            self.safety_check_count = 0

    def combine(self, other: SafetyMetrics) -> SafetyMetrics:
        """
        Combine metrics from another SafetyMetrics instance.

        Args:
            other (SafetyMetrics): Another metrics instance to combine with.

        Returns:
            SafetyMetrics: New instance with combined metrics.
        """
        # Use both locks in consistent order to prevent deadlocks
        locks = sorted([self._lock, other._lock], key=lambda x: id(x))
        with locks[0]:
            with locks[1]:
                return SafetyMetrics(
                    total_generations=self.total_generations + other.total_generations,
                    blocked_generations=self.blocked_generations + other.blocked_generations,
                    suppression_events=self.suppression_events + other.suppression_events,
                    cache_hits=self.cache_hits + other.cache_hits,
                    cache_misses=self.cache_misses + other.cache_misses,
                    total_safety_check_time_ms=self.total_safety_check_time_ms + other.total_safety_check_time_ms,
                    safety_check_count=self.safety_check_count + other.safety_check_count,
                )


class SafetyChecker(ABC):
    """
    Abstract base class for all safety checkers.

    Safety checkers are responsible for analyzing text content and detecting various types of safety violations
    such as toxicity, bias, personally identifiable information, or other harmful content.
    """

    @abstractmethod
    def check_safety(self, text: Union[str, list[str]], **kwargs) -> Union[SafetyResult, list[SafetyResult]]:
        """
        Check text(s) for safety violations.

        Args:
            text (`Union[str, List[str]]`):
                Single text string or list of texts to check for safety violations.
            **kwargs:
                Additional checker-specific parameters.

        Returns:
            `Union[SafetyResult, List[SafetyResult]]`:
                SafetyResult for single text input, List[SafetyResult] for multiple texts.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract class. Only classes inheriting this class can be called."
        )

    @property
    @abstractmethod
    def supported_categories(self) -> list[str]:
        """
        Return list of safety categories this checker supports.

        Returns:
            `List[str]`: List of supported safety categories (e.g., ["toxicity", "bias"]).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract class. Only classes inheriting this class can be called."
        )

    def get_config(self) -> dict[str, Any]:
        """
        Return checker configuration for serialization.

        Returns:
            `Dict[str, Any]`: Dictionary containing the checker's configuration parameters.
        """
        return {"checker_type": self.__class__.__name__}


@dataclass
class SafetyState:
    """
    Tracks incremental safety checking state for efficient sequence processing.

    This class maintains state information to enable efficient sliding window
    and incremental safety checking, avoiding redundant processing of previously
    checked content.

    Args:
        last_check_position (`int`, *optional*, defaults to `0`):
            The position (in tokens) where the last safety check ended.
        last_check_result (`Optional[SafetyResult]`, *optional*):
            The result of the last safety check performed.
        sequence_prefix (`str`, *optional*, defaults to `""`):
            The text prefix that has already been checked for safety.
        is_safe_so_far (`bool`, *optional*, defaults to `True`):
            Whether the sequence has been safe up to the last check position.
        window_start_position (`int`, *optional*, defaults to `0`):
            The starting position of the current sliding window.
    """

    last_check_position: int = 0
    last_check_result: Optional[SafetyResult] = None
    sequence_prefix: str = ""
    is_safe_so_far: bool = True
    window_start_position: int = 0

    def should_check_incremental(self, current_position: int, min_new_tokens: int = 5) -> bool:
        """
        Determine if an incremental safety check should be performed.

        Args:
            current_position (`int`):
                Current position in the sequence (in tokens).
            min_new_tokens (`int`, *optional*, defaults to `5`):
                Minimum number of new tokens before triggering a new check.

        Returns:
            `bool`: True if a new safety check should be performed.
        """
        # Always check if this is the first check
        if self.last_check_position == 0:
            return True

        # Check if enough new tokens have been added
        new_tokens = current_position - self.last_check_position
        return new_tokens >= min_new_tokens

    def update_check_result(self, position: int, result: SafetyResult, sequence_prefix: str = "") -> None:
        """
        Update the state with a new safety check result.

        Args:
            position (`int`):
                The position where this check ended.
            result (`SafetyResult`):
                The safety check result.
            sequence_prefix (`str`, *optional*, defaults to `""`):
                The sequence prefix that was checked.
        """
        self.last_check_position = position
        self.last_check_result = result
        self.sequence_prefix = sequence_prefix
        self.is_safe_so_far = result.is_safe if result else True

    def get_incremental_text(self, full_text: str, sliding_window_size: int = -1) -> tuple[str, int]:
        """
        Extract the portion of text that needs incremental checking.

        Args:
            full_text (`str`):
                The complete sequence text.
            sliding_window_size (`int`, *optional*, defaults to `-1`):
                Size of sliding window in characters. -1 means no sliding window.

        Returns:
            `tuple[str, int]`: The text portion to check and its start position.
        """
        if sliding_window_size == -1:
            # No sliding window - return text from last check position
            if len(self.sequence_prefix) > 0:
                # Find where we left off and return remaining text
                remaining_text = full_text[len(self.sequence_prefix) :]
                return self.sequence_prefix + remaining_text, 0
            return full_text, 0
        # Use sliding window
        if len(full_text) <= sliding_window_size:
            return full_text, 0
        window_start = max(0, len(full_text) - sliding_window_size)
        self.window_start_position = window_start
        return full_text[window_start:], window_start

    def reset(self) -> None:
        """Reset the safety state for a new sequence."""
        self.last_check_position = 0
        self.last_check_result = None
        self.sequence_prefix = ""
        self.is_safe_so_far = True
        self.window_start_position = 0
