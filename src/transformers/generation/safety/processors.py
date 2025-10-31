# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional

import torch

from ..logits_process import LogitsProcessor
from ..stopping_criteria import StoppingCriteria
from .base import SafetyChecker, SafetyMetrics, SafetyResult, SafetyState, SafetyViolation
from .configuration import SafetyConfig


logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_CACHE_SIZE = 100
DEFAULT_UNSAFE_HASH_LIMIT = 1000
DEFAULT_CHECK_INTERVAL = 1


class _SafetyCache:
    """Simple LRU cache for safety check results."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
        self.max_size = max_size
        self._cache = OrderedDict()

    def get(self, text: str, use_prefix_matching: bool = False):
        """
        Get cached result and move to end for LRU.

        Args:
            text: Text to look up (will be hashed to create cache key)
            use_prefix_matching: Ignored for simple cache (only supported by prefix cache)

        Returns:
            SafetyResult if found, None otherwise
        """
        key = _generate_cache_key(text)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return None

    def put(self, text: str, value) -> None:
        """
        Put result in cache with LRU eviction.

        Args:
            text: The text that was checked (will be hashed to create cache key)
            value: The SafetyResult to store
        """
        key = _generate_cache_key(text)
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __contains__(self, text: str) -> bool:
        """Check if text exists in cache."""
        key = _generate_cache_key(text)
        return key in self._cache


class _PrefixSafetyCache:
    """
    Advanced caching system that supports prefix-based caching for efficient sequence checking.

    This cache can reuse safety results for sequences that share common prefixes,
    significantly improving performance for incremental checking scenarios.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_CACHE_SIZE,
        prefix_lengths: Optional[list[int]] = None,
        min_text_length_for_prefix: int = 50,
    ):
        self.max_size = max_size
        self.prefix_lengths = prefix_lengths if prefix_lengths is not None else [100, 75, 50]
        self.min_text_length_for_prefix = min_text_length_for_prefix
        self._cache = OrderedDict()  # Maps full cache keys to results
        self._prefix_map = {}  # Maps text prefixes to cache keys that contain them

    def get(self, text: str, use_prefix_matching: bool = True):
        """
        Get cached result, optionally using prefix matching for efficiency.

        Args:
            text: Text to look up
            use_prefix_matching: Whether to try prefix matching if exact match fails

        Returns:
            SafetyResult if found, None otherwise
        """
        cache_key = _generate_cache_key(text)

        # Try exact match first
        if cache_key in self._cache:
            result = self._cache.pop(cache_key)
            self._cache[cache_key] = result  # Move to end for LRU
            return result

        # If prefix matching is enabled and exact match failed
        if use_prefix_matching:
            return self._try_prefix_match(text)

        return None

    def put(self, text: str, result) -> None:
        """
        Store result in cache with prefix indexing.

        Args:
            text: The text that was checked
            result: The SafetyResult to store
        """
        cache_key = _generate_cache_key(text)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            old_key, _ = self._cache.popitem(last=False)
            self._cleanup_prefix_references(old_key)

        # Store result
        self._cache[cache_key] = result

        # Update prefix mapping for common prefixes
        if len(text) > self.min_text_length_for_prefix:  # Only index prefixes for longer texts
            # Use the longest configured prefix length that's not larger than half the text
            max_prefix_length = max([length for length in self.prefix_lengths if length <= len(text) // 2], default=0)
            if max_prefix_length > 0:
                prefix = text[:max_prefix_length]
                prefix_key = _generate_cache_key(prefix)

                if prefix_key not in self._prefix_map:
                    self._prefix_map[prefix_key] = set()
                self._prefix_map[prefix_key].add(cache_key)

    def _try_prefix_match(self, text: str):
        """
        Try to find a cached result for a prefix of the given text.

        This is useful when we have cached results for shorter versions of the sequence.
        """
        if len(text) < self.min_text_length_for_prefix:  # Don't use prefix matching for very short texts
            return None

        # Try progressively shorter prefixes from configuration
        for prefix_len in sorted(self.prefix_lengths, reverse=True):
            if len(text) <= prefix_len:
                continue

            prefix = text[:prefix_len]
            prefix_key = _generate_cache_key(prefix)

            if prefix_key in self._prefix_map:
                # Found potential matches - check if any are safe
                for candidate_key in self._prefix_map[prefix_key]:
                    if candidate_key in self._cache:
                        result = self._cache[candidate_key]
                        # Only reuse if the cached result was safe
                        # (unsafe results might not apply to the longer sequence)
                        if result.is_safe:
                            # Move to end for LRU
                            self._cache.move_to_end(candidate_key)
                            return result

        return None

    def _cleanup_prefix_references(self, removed_cache_key: str) -> None:
        """Remove references to evicted cache keys from prefix mapping."""
        keys_to_remove = []
        for prefix_key, cache_keys in self._prefix_map.items():
            if removed_cache_key in cache_keys:
                cache_keys.discard(removed_cache_key)
                if not cache_keys:  # No more references
                    keys_to_remove.append(prefix_key)

        for key in keys_to_remove:
            del self._prefix_map[key]

    def __contains__(self, text: str) -> bool:
        """Check if text exists in cache."""
        cache_key = _generate_cache_key(text)
        return cache_key in self._cache


def _generate_cache_key(text: str) -> str:
    """
    Generate a SHA-256 based cache key for text content.

    Uses length prefix for quick rejection of different-sized texts,
    followed by SHA-256 hash for collision-resistant uniqueness.

    Args:
        text (str): The text content to generate a cache key for.

    Returns:
        str: A cache key in the format "length:hash"
    """
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{len(text)}:{text_hash}"


class _SlidingWindowSafetyMixin:
    """
    Shared functionality for sliding window safety processing.

    This mixin provides common methods for both SafetyLogitsProcessor and
    SafetyStoppingCriteria to handle sliding window text extraction,
    incremental checking, and cache management.
    """

    def _get_text_to_check(self, full_text: str, safety_state: SafetyState) -> tuple[str, int]:
        """
        Determine what text to check based on sliding window and incremental settings.

        Args:
            full_text: The complete sequence text
            safety_state: The safety state for this sequence

        Returns:
            tuple[str, int]: Text to check and window start position
        """
        if self.incremental_checking:
            # Use incremental checking with sliding window
            return safety_state.get_incremental_text(
                full_text, self.sliding_window_size if self.sliding_window_size > 0 else -1
            )
        # Use sliding window without incremental state
        if self.sliding_window_size > 0 and len(full_text) > self.sliding_window_size:
            # Extract sliding window (character-based approximation)
            text_to_check = full_text[-self.sliding_window_size :]
            window_start = len(full_text) - self.sliding_window_size
            return text_to_check, window_start
        return full_text, 0

    def _should_skip_safety_check(
        self, safety_state: SafetyState, current_position: int, min_new_tokens: int = 5
    ) -> tuple[bool, SafetyResult]:
        """
        Determine if we should skip the safety check and return cached result.

        Args:
            safety_state: The safety state for this sequence
            current_position: Current position in tokens
            min_new_tokens: Minimum tokens required for new check

        Returns:
            tuple[bool, SafetyResult]: Whether to skip check and result to use if skipping
        """
        if not self.incremental_checking:
            return False, None

        if not safety_state.should_check_incremental(current_position, min_new_tokens):
            # Use previous result if available
            safety_result = safety_state.last_check_result
            if safety_result is not None:
                return True, safety_result
        return False, None

    def _perform_safety_check(self, text_to_check: str) -> SafetyResult:
        """
        Perform safety check with caching and error handling.

        Args:
            text_to_check: Text to check for safety violations

        Returns:
            SafetyResult: Result of the safety check
        """
        # Check cache first (unified interface)
        safety_result = self._sequence_cache.get(text_to_check, use_prefix_matching=True)

        if safety_result is not None:
            self._metrics.record_cache_hit()
            return safety_result

        # Check safety of text portion - record timing
        start_time = time.perf_counter()
        try:
            safety_result = self.safety_checker.check_safety(text_to_check)
        except Exception as e:
            logger.warning(f"Safety check failed for sequence: {e}")
            # Fail safe - assume unsafe if we can't check
            safety_result = SafetyResult(
                is_safe=False,
                confidence=0.0,
                violations=[SafetyViolation("unknown", 0.0, "high", "Safety check failed")],
                metadata={"error": str(e)},
            )

        # Record timing and cache miss
        end_time = time.perf_counter()
        check_time_ms = (end_time - start_time) * 1000
        self._metrics.record_safety_check(check_time_ms)
        self._metrics.record_cache_miss()

        # Cache the result
        self._sequence_cache.put(text_to_check, safety_result)
        return safety_result

    def _update_safety_state(
        self,
        safety_state: SafetyState,
        current_position: int,
        safety_result: SafetyResult,
        text_to_check: str,
        window_start: int,
        full_text: str,
    ) -> None:
        """
        Update safety state with new check result if using incremental checking.

        Args:
            safety_state: The safety state to update
            current_position: Current position in sequence
            safety_result: Result from safety check
            text_to_check: Text that was checked
            window_start: Start position of the window
            full_text: Complete sequence text
        """
        if self.incremental_checking:
            safety_state.update_check_result(
                current_position, safety_result, text_to_check if window_start == 0 else full_text
            )


class SafetyLogitsProcessor(LogitsProcessor, _SlidingWindowSafetyMixin):
    """
    [`LogitsProcessor`] that blocks generation when unsafe content is detected.

    This processor checks the current sequence for safety violations and blocks
    further generation by suppressing all tokens when unsafe content is detected.
    It integrates with the transformers safety framework to provide real-time
    content blocking.

    Args:
        safety_checker ([`SafetyChecker`]):
            The safety checker to use for content evaluation.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for decoding sequences.
        safety_config ([`SafetyConfig`]):
            Configuration for safety checking.
        check_interval (`int`, *optional*, defaults to 1):
            Check safety every N tokens. Must be positive.
        suppress_threshold (`float`, *optional*, defaults to negative infinity):
            Logit value for suppressing unsafe tokens.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from transformers.generation.safety import SafetyLogitsProcessor, SafetyConfig
    >>> from examples.safe_generation import BasicToxicityChecker

    >>> # Initialize model and tokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token

    >>> # Create safety checker and config
    >>> safety_checker = BasicToxicityChecker()
    >>> safety_config = SafetyConfig.from_checker(safety_checker)
    >>> safety_processor = SafetyLogitsProcessor(
    ...     safety_checker=safety_checker,
    ...     tokenizer=tokenizer,
    ...     safety_config=safety_config
    ... )

    >>> # Generate with safety filtering
    >>> inputs = tokenizer("Tell me about", return_tensors="pt")
    >>> outputs = model.generate(
    ...     **inputs,
    ...     logits_processor=[safety_processor],
    ...     max_new_tokens=50,
    ...     do_sample=True
    ... )
    >>> generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```
    """

    def __init__(
        self,
        safety_checker: SafetyChecker,
        tokenizer,
        safety_config: SafetyConfig,
        check_interval: int = 1,
        suppress_threshold: float = -float("inf"),
    ):
        """
        Initialize the SafetyLogitsProcessor.

        Args:
            safety_checker: The safety checker to use for content evaluation
            tokenizer: The tokenizer used for decoding sequences
            safety_config: Configuration for safety checking
            check_interval: Check safety every N tokens (default: 1, must be positive)
            suppress_threshold: Logit value for suppressing unsafe tokens

        Raises:
            ValueError: If check_interval is not positive
        """
        # Input validation
        if not isinstance(check_interval, int) or check_interval < 1:
            raise ValueError(f"check_interval must be a positive integer, got {check_interval}")

        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.safety_config = safety_config
        self.check_interval = check_interval
        self.suppress_threshold = suppress_threshold
        self._step_count = 0

        # Initialize sliding window and incremental checking
        self._safety_states = {}  # Track safety state per sequence in the batch
        self.sliding_window_size = getattr(safety_config, "sliding_window_size", 512)
        self.incremental_checking = getattr(safety_config, "incremental_checking", True)

        # Initialize cache with configured size (use prefix cache if incremental checking is enabled)
        cache_size = getattr(safety_config, "cache_size", DEFAULT_CACHE_SIZE)
        if self.incremental_checking:
            prefix_lengths = getattr(safety_config, "prefix_lengths", [100, 75, 50])
            min_text_length_for_prefix = getattr(safety_config, "min_text_length_for_prefix", 50)
            self._sequence_cache = _PrefixSafetyCache(
                max_size=cache_size,
                prefix_lengths=prefix_lengths,
                min_text_length_for_prefix=min_text_length_for_prefix,
            )  # Advanced prefix-aware cache
        else:
            self._sequence_cache = _SafetyCache(max_size=cache_size)  # Simple LRU cache
        self._metrics = SafetyMetrics()  # Initialize metrics collection

    def _apply_token_suppression(self, scores: torch.FloatTensor, batch_idx: int, safety_result: SafetyResult) -> None:
        """
        Apply token suppression for unsafe content.

        Args:
            scores: Token scores tensor to modify
            batch_idx: Index in the batch
            safety_result: Safety check result
        """
        if not safety_result.is_safe:
            tokens_to_suppress = self._get_tokens_to_suppress(scores[batch_idx], safety_result)
            if len(tokens_to_suppress) > 0:
                device = scores.device
                if isinstance(tokens_to_suppress, list):
                    tokens_to_suppress = torch.tensor(tokens_to_suppress, device=device)
                scores[batch_idx, tokens_to_suppress] = self.suppress_threshold
                self._metrics.record_suppression_event()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply safety filtering to token scores.

        Args:
            input_ids: Current sequence tokens [batch_size, seq_len]
            scores: Token logits [batch_size, vocab_size]

        Returns:
            Modified scores with unsafe tokens suppressed
        """
        self._step_count += 1

        # Only check safety at specified intervals for performance
        if self._step_count % self.check_interval != 0:
            return scores

        batch_size = input_ids.shape[0]
        modified_scores = scores.clone()

        for i in range(batch_size):
            # Convert current sequence to text
            current_sequence = input_ids[i]
            full_text = self.tokenizer.decode(current_sequence, skip_special_tokens=True)

            # Get or create safety state for this sequence
            seq_id = i
            if seq_id not in self._safety_states:
                self._safety_states[seq_id] = SafetyState()
            safety_state = self._safety_states[seq_id]

            # Determine what text to check based on configuration
            text_to_check, window_start = self._get_text_to_check(full_text, safety_state)
            current_position = len(current_sequence)

            # Check if we can skip the safety check and reuse previous result
            should_skip, cached_result = self._should_skip_safety_check(safety_state, current_position)
            if should_skip:
                self._apply_token_suppression(modified_scores, i, cached_result)
                continue

            # Perform safety check with caching and error handling
            safety_result = self._perform_safety_check(text_to_check)

            # Update safety state if using incremental checking
            self._update_safety_state(
                safety_state, current_position, safety_result, text_to_check, window_start, full_text
            )

            # Apply suppression if content is unsafe
            self._apply_token_suppression(modified_scores, i, safety_result)

            # Debug logging for unsafe content
            if not safety_result.is_safe:
                logger.debug(f"Suppressed tokens using blocking strategy for sequence: {full_text[:50]}...")

        return modified_scores

    def _get_tokens_to_suppress(self, scores: torch.FloatTensor, safety_result) -> torch.Tensor:
        """
        Determine which tokens to suppress when unsafe content is detected.

        Args:
            scores: Token logits for a single sequence [vocab_size]
            safety_result: The safety assessment result (unused for blocking strategy)

        Returns:
            Tensor of token indices to suppress (all tokens for blocking)
        """
        # Block strategy: suppress all tokens to force generation to stop
        return torch.arange(scores.size(0), device=scores.device)

    def get_metrics(self) -> SafetyMetrics:
        """
        Return current metrics for this processor.

        Returns:
            SafetyMetrics: Current metrics collection.
        """
        return self._metrics

    def reset_safety_states(self) -> None:
        """
        Reset all safety states. Call this when starting a new generation batch.
        """
        self._safety_states.clear()

    def _get_text_for_safety_check(self, full_text: str, safety_state: SafetyState) -> tuple[str, int]:
        """
        Extract the appropriate text portion for safety checking.

        Args:
            full_text: The complete sequence text
            safety_state: Current safety state for incremental checking

        Returns:
            tuple[str, int]: Text to check and its starting position
        """
        if self.incremental_checking:
            return safety_state.get_incremental_text(
                full_text, self.sliding_window_size if self.sliding_window_size > 0 else -1
            )
        # Simple sliding window without incremental state
        if self.sliding_window_size > 0 and len(full_text) > self.sliding_window_size:
            window_start = len(full_text) - self.sliding_window_size
            return full_text[window_start:], window_start
        return full_text, 0


class SafetyStoppingCriteria(StoppingCriteria, _SlidingWindowSafetyMixin):
    """
    [`StoppingCriteria`] that halts generation when unsafe content is detected.

    This provides a sequence-level safety check that can stop generation before
    unsafe content is returned to the user. It works as a final safety gate
    after token-level filtering by SafetyLogitsProcessor.

    Args:
        safety_checker ([`SafetyChecker`]):
            The safety checker to use for content evaluation.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for decoding sequences.
        safety_config ([`SafetyConfig`]):
            Configuration for safety checking.
        check_final_only (`bool`, *optional*, defaults to `False`):
            If True, only check safety on the final call (when all sequences are complete).
            If False, check safety on every call during generation.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from transformers.generation.safety import SafetyStoppingCriteria, SafetyConfig
    >>> from examples.safe_generation import BasicToxicityChecker

    >>> # Initialize model and tokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> tokenizer.pad_token = tokenizer.eos_token

    >>> # Create safety checker and config
    >>> safety_checker = BasicToxicityChecker()
    >>> safety_config = SafetyConfig.from_checker(safety_checker)
    >>> safety_stopping = SafetyStoppingCriteria(
    ...     safety_checker=safety_checker,
    ...     tokenizer=tokenizer,
    ...     safety_config=safety_config
    ... )

    >>> # Generate with safety stopping
    >>> inputs = tokenizer("Tell me about", return_tensors="pt")
    >>> outputs = model.generate(
    ...     **inputs,
    ...     stopping_criteria=[safety_stopping],
    ...     max_new_tokens=50,
    ...     do_sample=True
    ... )
    >>> generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ```
    """

    def __init__(
        self, safety_checker: SafetyChecker, tokenizer, safety_config: SafetyConfig, check_final_only: bool = False
    ):
        """
        Initialize the SafetyStoppingCriteria.

        Args:
            safety_checker: The safety checker to use for content evaluation
            tokenizer: The tokenizer used for decoding sequences
            safety_config: Configuration for safety checking
            check_final_only: If True, only check when generation is complete

        Raises:
            ValueError: If safety_checker is None
        """
        if safety_checker is None:
            raise ValueError("safety_checker cannot be None")

        self.safety_checker = safety_checker
        self.tokenizer = tokenizer
        self.safety_config = safety_config
        self.check_final_only = check_final_only
        self._unsafe_sequence_hashes = OrderedDict()  # Track unsafe sequences by content hash (LRU)

        # Initialize sliding window and incremental checking
        self._safety_states = {}  # Track safety state per sequence in the batch
        self.sliding_window_size = getattr(safety_config, "sliding_window_size", 512)
        self.incremental_checking = getattr(safety_config, "incremental_checking", True)

        # Initialize cache with configured size (use prefix cache if incremental checking is enabled)
        cache_size = getattr(safety_config, "cache_size", DEFAULT_CACHE_SIZE)
        if self.incremental_checking:
            prefix_lengths = getattr(safety_config, "prefix_lengths", [100, 75, 50])
            min_text_length_for_prefix = getattr(safety_config, "min_text_length_for_prefix", 50)
            self._sequence_cache = _PrefixSafetyCache(
                max_size=cache_size,
                prefix_lengths=prefix_lengths,
                min_text_length_for_prefix=min_text_length_for_prefix,
            )  # Advanced prefix-aware cache
        else:
            self._sequence_cache = _SafetyCache(max_size=cache_size)  # Simple LRU cache
        # Get configured unsafe hash limit
        self._unsafe_hash_limit = getattr(safety_config, "unsafe_hash_limit", DEFAULT_UNSAFE_HASH_LIMIT)
        self._metrics = SafetyMetrics()  # Initialize metrics collection

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        """
        Check if generation should stop due to safety violations.

        Args:
            input_ids: Current sequences [batch_size, seq_len]
            scores: Token scores [batch_size, vocab_size]

        Returns:
            Boolean tensor indicating which sequences should stop [batch_size]
        """
        batch_size = input_ids.shape[0]

        # Record generation attempts for metrics
        for _ in range(batch_size):
            self._metrics.record_generation_attempt()

        # Initialize should_stop tensor
        should_stop = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # If check_final_only is True, skip safety checks during generation
        # This will be handled by other safety mechanisms or post-generation checks
        if self.check_final_only and not kwargs.get("is_final_call", False):
            return should_stop

        # Check each sequence for safety violations
        for i in range(batch_size):
            # Convert current sequence to text
            current_sequence = input_ids[i]
            full_text = self.tokenizer.decode(current_sequence, skip_special_tokens=True)

            # Get or create safety state for this sequence
            seq_id = i
            if seq_id not in self._safety_states:
                self._safety_states[seq_id] = SafetyState()
            safety_state = self._safety_states[seq_id]

            # Determine what text to check based on configuration
            text_to_check, window_start = self._get_text_to_check(full_text, safety_state)
            current_position = len(current_sequence)

            # Check if we can skip the safety check and reuse previous result (less aggressive for stopping)
            should_skip, cached_result = self._should_skip_safety_check(
                safety_state, current_position, min_new_tokens=3
            )
            if should_skip and cached_result is not None and not cached_result.is_safe:
                should_stop[i] = True
                continue

            # Create sequence hash for tracking (using consistent SHA-256 hash)
            sequence_hash = _generate_cache_key(text_to_check)

            # Check if this sequence content is already known to be unsafe (LRU)
            if sequence_hash in self._unsafe_sequence_hashes:
                # Move to end for LRU behavior
                self._unsafe_sequence_hashes.move_to_end(sequence_hash)
                should_stop[i] = True
                continue

            # Perform safety check with caching and error handling
            safety_result = self._perform_safety_check(text_to_check)

            # Update safety state if using incremental checking
            self._update_safety_state(
                safety_state, current_position, safety_result, text_to_check, window_start, full_text
            )

            # If unsafe, mark sequence for stopping
            if not safety_result.is_safe:
                # Add to unsafe hash dict with LRU eviction to prevent memory leaks
                if len(self._unsafe_sequence_hashes) >= self._unsafe_hash_limit:
                    # Remove oldest entry (LRU eviction)
                    self._unsafe_sequence_hashes.popitem(last=False)
                # Add new entry (will be at the end = most recently used)
                self._unsafe_sequence_hashes[sequence_hash] = True  # Track by content hash
                should_stop[i] = True
                self._metrics.record_blocked_generation()

                # Log safety violation for debugging
                violation_categories = [v.category for v in safety_result.violations]
                logger.warning(
                    f"Generation stopped for sequence {i} due to safety violations: {violation_categories}. "
                    f"Text: {full_text[:100]}..."
                )

        return should_stop

    def get_metrics(self) -> SafetyMetrics:
        """
        Return current metrics for this stopping criteria.

        Returns:
            SafetyMetrics: Current metrics collection.
        """
        return self._metrics

    def reset_safety_states(self) -> None:
        """
        Reset all safety states. Call this when starting a new generation batch.
        """
        self._safety_states.clear()
