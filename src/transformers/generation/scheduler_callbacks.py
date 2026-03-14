# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Preset scheduler callbacks for common generation intervention patterns.

These callbacks demonstrate how to use the `GenerationScheduler` callback system
and provide ready-to-use implementations for common use cases:

    - `EntropyMonitorCallback`: Monitors token entropy and can pause on high uncertainty.
    - `TokenPatternCallback`: Detects token patterns (e.g., tool calls) and pauses.
    - `GenerationLoggerCallback`: Logs generation progress for debugging.
    - `StepBudgetCallback`: Enforces a maximum number of decoding steps.
"""

from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional as F

from ..utils import logging
from .generation_scheduler import ControlTokenParser, SchedulerCallback, SchedulerContext
from .state_machine import GenerationPhase, GenerationState, SchedulerMode


logger = logging.get_logger(__name__)


class EntropyMonitorCallback(SchedulerCallback):
    """
    Monitors the entropy of the token probability distribution.

    Can pause generation when entropy exceeds a threshold (indicating
    model uncertainty), enabling external intervention like retrieval
    augmentation or context injection.

    Args:
        entropy_threshold (`float`, defaults to 4.0):
            Entropy threshold above which to pause generation.
        window_size (`int`, defaults to 5):
            Number of recent entropy values to track for moving average.
        min_step (`int`, defaults to 1):
            Minimum step before entropy checking begins.
        action (`str`, defaults to `"pause"`):
            Action to take when threshold is exceeded: "pause" or "log".

    Example:
        ```python
        callback = EntropyMonitorCallback(entropy_threshold=3.5, action="pause")
        scheduler.register_callback(callback)
        output = model.generate(input_ids, scheduler=scheduler)
        # Check if paused due to high entropy
        if scheduler.is_paused():
            print(f"High entropy at step {scheduler.get_state().step}")
            print(f"Entropy history: {callback.entropy_history}")
        ```
    """

    def __init__(
        self,
        entropy_threshold: float = 4.0,
        window_size: int = 5,
        min_step: int = 1,
        action: str = "pause",
    ):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.min_step = min_step
        self.action = action
        self.entropy_history: list[float] = []
        self._window: deque = deque(maxlen=window_size)

    def _compute_entropy(self, logits: torch.FloatTensor) -> float:
        """Compute the Shannon entropy of the logits distribution."""
        probs = F.softmax(logits, dim=-1)
        # Clamp to avoid log(0)
        log_probs = torch.log(probs.clamp(min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        return entropy

    def on_logits_ready(
        self,
        logits: torch.FloatTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> torch.FloatTensor:
        """Compute and record entropy from logits."""
        entropy = self._compute_entropy(logits)
        self.entropy_history.append(entropy)
        self._window.append(entropy)

        # Store in state metadata for external access
        state.metadata["current_entropy"] = entropy
        state.metadata["mean_entropy"] = sum(self._window) / len(self._window)

        return logits

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Check entropy threshold and optionally pause."""
        if state.step < self.min_step:
            return True

        if not self.entropy_history:
            return True

        current_entropy = self.entropy_history[-1]

        if current_entropy > self.entropy_threshold:
            if self.action == "pause":
                context.should_pause = True
                context.custom_data["pause_reason"] = "high_entropy"
                context.custom_data["entropy_at_pause"] = current_entropy
                logger.info(
                    f"EntropyMonitor: entropy={current_entropy:.3f} > threshold={self.entropy_threshold:.3f} "
                    f"at step {state.step}, pausing generation"
                )
                return False
            else:
                logger.info(
                    f"EntropyMonitor: entropy={current_entropy:.3f} > threshold={self.entropy_threshold:.3f} "
                    f"at step {state.step}"
                )

        return True

    @property
    def mean_entropy(self) -> float:
        """Moving average of recent entropy values."""
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)


class TokenPatternCallback(SchedulerCallback):
    """
    Detects specific token patterns in the generated sequence and pauses.

    Useful for detecting tool calls, function invocations, or other
    structured output markers during generation.

    Args:
        trigger_token_ids (`Set[int]`, *optional*):
            Set of individual token IDs that trigger a pause.
        trigger_sequences (`List[List[int]]`, *optional*):
            List of token ID sequences that trigger a pause when matched.
        on_trigger (`callable`, *optional*):
            Custom function called when a pattern is detected:
            `(token_id, state, context) -> bool`. Return False to pause.

    Example:
        ```python
        # Pause when specific tokens are generated
        callback = TokenPatternCallback(
            trigger_token_ids={tokenizer.encode("<tool_call>")[0]}
        )
        scheduler.register_callback(callback)
        ```
    """

    def __init__(
        self,
        trigger_token_ids: set[int] | None = None,
        trigger_sequences: list[list[int]] | None = None,
        on_trigger: Callable | None = None,
    ):
        super().__init__()
        self.trigger_token_ids = trigger_token_ids or set()
        self.trigger_sequences = trigger_sequences or []
        self.on_trigger = on_trigger
        self.triggered_at: list[int] = []  # Steps where triggers occurred
        self._recent_tokens: deque = deque(
            maxlen=max((len(seq) for seq in self.trigger_sequences), default=1)
        )

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Check if the generated token matches any trigger pattern."""
        # Handle batched token_ids (take first sequence)
        if token_id.dim() > 0:
            tid = token_id[0].item()
        else:
            tid = token_id.item()

        self._recent_tokens.append(tid)

        # Check individual token triggers
        if tid in self.trigger_token_ids:
            self.triggered_at.append(state.step)
            context.custom_data["trigger_type"] = "token_id"
            context.custom_data["trigger_token"] = tid

            if self.on_trigger is not None:
                return self.on_trigger(token_id, state, context)

            context.should_pause = True
            logger.info(f"TokenPattern: trigger token {tid} detected at step {state.step}")
            return False

        # Check sequence triggers
        recent_list = list(self._recent_tokens)
        for seq in self.trigger_sequences:
            if len(recent_list) >= len(seq) and recent_list[-len(seq):] == seq:
                self.triggered_at.append(state.step)
                context.custom_data["trigger_type"] = "sequence"
                context.custom_data["trigger_sequence"] = seq

                if self.on_trigger is not None:
                    return self.on_trigger(token_id, state, context)

                context.should_pause = True
                logger.info(f"TokenPattern: trigger sequence {seq} detected at step {state.step}")
                return False

        return True


class GenerationLoggerCallback(SchedulerCallback):
    """
    Logs generation progress for debugging and observability.

    Records phase transitions, token generation events, and timing
    information. Useful for understanding the generation process.

    Args:
        log_tokens (`bool`, defaults to `True`):
            Whether to log individual token generation events.
        log_phases (`bool`, defaults to `True`):
            Whether to log phase transitions.
        log_logits_stats (`bool`, defaults to `False`):
            Whether to log logits statistics (mean, std, max).
        tokenizer (*optional*):
            If provided, logs decoded text alongside token IDs.

    Example:
        ```python
        callback = GenerationLoggerCallback(tokenizer=tokenizer)
        scheduler.register_callback(callback)
        output = model.generate(input_ids, scheduler=scheduler)
        print(callback.get_log())
        ```
    """

    def __init__(
        self,
        log_tokens: bool = True,
        log_phases: bool = True,
        log_logits_stats: bool = False,
        tokenizer=None,
    ):
        super().__init__()
        self.log_tokens = log_tokens
        self.log_phases = log_phases
        self.log_logits_stats = log_logits_stats
        self.tokenizer = tokenizer
        self._log: list[dict[str, Any]] = []

    def on_phase_transition(
        self,
        from_phase: GenerationPhase,
        to_phase: GenerationPhase,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        if self.log_phases:
            entry = {
                "type": "phase_transition",
                "from": from_phase.name,
                "to": to_phase.name,
                "step": state.step,
                "timestamp": state.timestamp,
            }
            self._log.append(entry)
            logger.info(f"GenerationLogger: {from_phase.name} → {to_phase.name} (step={state.step})")
        return True

    def on_logits_ready(
        self,
        logits: torch.FloatTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> torch.FloatTensor:
        if self.log_logits_stats:
            entry = {
                "type": "logits_stats",
                "step": state.step,
                "mean": logits.mean().item(),
                "std": logits.std().item(),
                "max": logits.max().item(),
                "min": logits.min().item(),
            }
            self._log.append(entry)
        return logits

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        if self.log_tokens:
            tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()
            entry = {
                "type": "token_generated",
                "step": state.step,
                "token_id": tid,
            }
            if self.tokenizer is not None:
                try:
                    entry["token_text"] = self.tokenizer.decode([tid])
                except Exception:
                    entry["token_text"] = "<decode_error>"
            self._log.append(entry)
        return True

    def on_generation_complete(self, state: GenerationState):
        entry = {
            "type": "generation_complete",
            "total_steps": state.step,
            "final_length": state.input_ids.shape[-1] if state.input_ids is not None else 0,
        }
        self._log.append(entry)
        logger.info(f"GenerationLogger: complete (steps={state.step})")

    def on_error(self, error: Exception, state: GenerationState):
        entry = {
            "type": "error",
            "step": state.step,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        self._log.append(entry)
        logger.error(f"GenerationLogger: error at step {state.step}: {error}")

    def get_log(self) -> list[dict[str, Any]]:
        """Get the full generation log."""
        return list(self._log)

    def clear_log(self):
        """Clear the generation log."""
        self._log.clear()


class StepBudgetCallback(SchedulerCallback):
    """
    Enforces a maximum number of decoding steps.

    Unlike `max_length` in `GenerationConfig` (which counts total tokens
    including the prompt), this callback counts only the decoding steps.

    Args:
        max_steps (`int`):
            Maximum number of decoding steps to allow.
        warn_at (`float`, defaults to 0.9):
            Fraction of budget at which to log a warning.

    Example:
        ```python
        callback = StepBudgetCallback(max_steps=100)
        scheduler.register_callback(callback)
        output = model.generate(input_ids, scheduler=scheduler)
        ```
    """

    def __init__(self, max_steps: int, warn_at: float = 0.9):
        super().__init__()
        self.max_steps = max_steps
        self.warn_at = warn_at
        self._warned = False

    def on_step_end(
        self,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        if not self._warned and state.step >= int(self.max_steps * self.warn_at):
            logger.warning(
                f"StepBudget: {state.step}/{self.max_steps} steps used "
                f"({state.step / self.max_steps * 100:.0f}%)"
            )
            self._warned = True

        if state.step >= self.max_steps:
            context.should_pause = True
            context.custom_data["pause_reason"] = "step_budget_exceeded"
            logger.info(f"StepBudget: budget of {self.max_steps} steps exhausted")
            return False

        return True


class RepetitionDetectorCallback(SchedulerCallback):
    """
    Detects repetitive token patterns and can pause generation.

    Monitors the generated sequence for repetitive n-gram patterns.
    When repetition is detected above a threshold, it can pause
    generation for external intervention.

    Args:
        ngram_size (`int`, defaults to 3):
            Size of n-grams to monitor for repetition.
        max_repetitions (`int`, defaults to 3):
            Maximum allowed repetitions of the same n-gram before triggering.
        action (`str`, defaults to `"pause"`):
            Action to take: "pause" or "log".

    Example:
        ```python
        callback = RepetitionDetectorCallback(ngram_size=3, max_repetitions=2)
        scheduler.register_callback(callback)
        ```
    """

    def __init__(
        self,
        ngram_size: int = 3,
        max_repetitions: int = 3,
        action: str = "pause",
    ):
        super().__init__()
        self.ngram_size = ngram_size
        self.max_repetitions = max_repetitions
        self.action = action
        self._ngram_counts: dict[tuple, int] = {}
        self._recent_tokens: deque = deque(maxlen=ngram_size * (max_repetitions + 1))

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()
        self._recent_tokens.append(tid)

        if len(self._recent_tokens) >= self.ngram_size:
            recent = list(self._recent_tokens)
            ngram = tuple(recent[-self.ngram_size:])
            self._ngram_counts[ngram] = self._ngram_counts.get(ngram, 0) + 1

            if self._ngram_counts[ngram] >= self.max_repetitions:
                if self.action == "pause":
                    context.should_pause = True
                    context.custom_data["pause_reason"] = "repetition_detected"
                    context.custom_data["repeated_ngram"] = list(ngram)
                    context.custom_data["repetition_count"] = self._ngram_counts[ngram]
                    logger.info(
                        f"RepetitionDetector: n-gram {ngram} repeated "
                        f"{self._ngram_counts[ngram]} times at step {state.step}"
                    )
                    return False
                else:
                    logger.info(
                        f"RepetitionDetector: n-gram {ngram} repeated "
                        f"{self._ngram_counts[ngram]} times at step {state.step}"
                    )

        return True

    def reset_counts(self):
        """Reset n-gram counts."""
        self._ngram_counts.clear()
        self._recent_tokens.clear()


class StreamingSchedulerCallback(SchedulerCallback):
    """
    Bridges the scheduler with text streaming, yielding decoded tokens via a callback.

    This callback enables streaming output while using the scheduler. It invokes a
    user-provided ``on_text`` function each time a new token is generated, providing
    the decoded text chunk. This integrates with ``TextIteratorStreamer`` or any
    custom streaming pipeline.

    Args:
        tokenizer: A PreTrainedTokenizer for decoding token IDs.
        on_text (`callable`, *optional*):
            Function called with each decoded text chunk: ``(text: str) -> None``.
            If not provided, decoded tokens are accumulated in ``self.generated_text``.
        skip_special_tokens (`bool`, defaults to `True`):
            Whether to skip special tokens when decoding.
        skip_prompt (`bool`, defaults to `True`):
            Whether to skip the prompt tokens (only stream new tokens).

    Example:
        ```python
        tokens = []
        callback = StreamingSchedulerCallback(
            tokenizer=tokenizer,
            on_text=lambda text: tokens.append(text),
        )
        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(callback)
        output = model.generate(input_ids, scheduler=scheduler, max_new_tokens=50)
        print("".join(tokens))
        ```
    """

    def __init__(
        self,
        tokenizer,
        on_text: Callable | None = None,
        skip_special_tokens: bool = True,
        skip_prompt: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.skip_special_tokens = skip_special_tokens
        self.skip_prompt = skip_prompt
        self.generated_text: str = ""
        self._token_buffer: list[int] = []
        self._prompt_length: int = 0

    def on_phase_transition(
        self,
        from_phase: GenerationPhase,
        to_phase: GenerationPhase,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Record prompt length when entering DECODING phase."""
        if to_phase == GenerationPhase.DECODING and from_phase == GenerationPhase.PREFILL:
            if state.input_ids is not None:
                self._prompt_length = state.input_ids.shape[-1]
        return True

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Decode and stream each generated token."""
        tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()
        self._token_buffer.append(tid)

        try:
            text = self.tokenizer.decode(
                self._token_buffer,
                skip_special_tokens=self.skip_special_tokens,
            )
            # Only yield the new part (incremental decoding)
            new_text = text[len(self.generated_text):]
            if new_text:
                self.generated_text = text
                if self.on_text is not None:
                    self.on_text(new_text)
        except Exception:
            pass  # Silently ignore decode errors for partial tokens

        return True

    def get_text(self) -> str:
        """Get the full generated text so far."""
        return self.generated_text

    def reset(self):
        """Reset the streaming state."""
        self.generated_text = ""
        self._token_buffer.clear()
        self._prompt_length = 0


class InternalSchedulerCallback(SchedulerCallback):
    """
    Callback for INTERNAL mode — intercepts control tokens generated by the LLM.

    This callback works with a ``ControlTokenParser`` to detect when the model
    generates special control tokens (e.g., ``<STATE:READ_NEXT_CHUNK>``,
    ``<RECALL:chunk_3>``) and converts them into scheduler actions.

    It implements the ``token → control signal`` bridge described in the
    configurable generation modes design.

    Args:
        control_token_parser (`ControlTokenParser`):
            Parser that maps token IDs to actions.
        max_consecutive_controls (`int`, defaults to 10):
            Maximum consecutive control tokens before forcing a stop
            (safety guard against infinite loops).
        on_control_detected (`callable`, *optional*):
            Custom handler called when a control token is detected:
            ``(action_name, token_id, state, context) -> None``

    Example:
        ```python
        parser = ControlTokenParser(
            control_tokens={32000: "read_chunk", 32001: "summary_done", 32002: "recall"},
            action_handlers={
                "read_chunk": lambda name, state, ctx: ctx.custom_data.update({"action": "read_chunk"}),
                "summary_done": lambda name, state, ctx: setattr(ctx, 'should_pause', True),
            }
        )
        callback = InternalSchedulerCallback(control_token_parser=parser)
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        scheduler.register_callback(callback)
        ```
    """

    def __init__(
        self,
        control_token_parser: ControlTokenParser,
        max_consecutive_controls: int = 10,
        on_control_detected: Callable | None = None,
    ):
        super().__init__()
        self.parser = control_token_parser
        self.max_consecutive_controls = max_consecutive_controls
        self.on_control_detected = on_control_detected
        self._consecutive_control_count: int = 0
        self.control_history: list[dict[str, Any]] = []

    def on_token_generated(
        self,
        token_id: torch.LongTensor,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Intercept control tokens and execute associated actions."""
        # Only active in INTERNAL mode
        if context.mode != SchedulerMode.INTERNAL:
            return True

        tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()

        if self.parser.is_control_token(tid):
            action_name = self.parser.get_action(tid)
            self._consecutive_control_count += 1

            # Record in history
            self.control_history.append({
                "step": state.step,
                "token_id": tid,
                "action": action_name,
            })

            # Safety guard: prevent infinite control token loops
            if self._consecutive_control_count >= self.max_consecutive_controls:
                context.should_pause = True
                context.custom_data["pause_reason"] = "max_consecutive_controls"
                logger.warning(
                    f"InternalScheduler: {self._consecutive_control_count} consecutive control tokens "
                    f"detected at step {state.step}, forcing pause"
                )
                return False

            # Execute the action
            self.parser.execute(tid, state, context)

            # Call custom handler if provided
            if self.on_control_detected is not None:
                self.on_control_detected(action_name, tid, state, context)

            logger.debug(
                f"InternalScheduler: control token {tid} → action '{action_name}' at step {state.step}"
            )

            # If the action set should_pause, respect it
            if context.should_pause:
                return False

            return True
        else:
            # Reset consecutive counter on normal token
            self._consecutive_control_count = 0
            return True

    def on_step_begin(
        self,
        state: GenerationState,
        context: SchedulerContext,
    ) -> bool:
        """Reset per-step state."""
        return True

    def get_control_history(self) -> list[dict[str, Any]]:
        """Get the history of control token detections."""
        return list(self.control_history)

    def clear_history(self):
        """Clear the control history."""
        self.control_history.clear()
        self._consecutive_control_count = 0
