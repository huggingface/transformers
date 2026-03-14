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
Generation State Machine — Explicit state management for the transformers `generate()` loop.

This module introduces a formal state machine to replace the implicit state tracking
scattered across local variables in the generation loop. It aligns with the goals
outlined in Issue #30810 (decomposing `generate()` into composable stages).

Key components:
    - `SchedulerMode`: Enum of the 3 scheduler modes (NONE, INTERNAL, FORCE).
    - `GenerationPhase`: Enum of the 8 explicit generation phases.
    - `GenerationState`: A serializable snapshot of the full generation state at any point.
    - `GenerationStateMachine`: Manages phase transitions with validation.

Design principles:
    - Zero overhead when not used (empty callbacks, no extra copies).
    - 100% backward compatible — purely opt-in.
    - Serializable state for pause/resume/checkpoint.
    - Compatible with torch.compile (no graph breaks in the hot path).

Scheduler modes:
    - `NONE` (default): Plain generation, 100% original HF behavior.
    - `INTERNAL`: LLM-driven scheduling — the model generates control tokens
      that the scheduler interprets to drive state transitions.
    - `FORCE`: External scheduling — external code controls the generation
      pipeline via the scheduler API.
"""

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from ..utils import logging


logger = logging.get_logger(__name__)


class SchedulerMode(Enum):
    """
    Scheduler operating modes for the generation pipeline.

    Determines how (and whether) the scheduler participates in generation.

    Modes:
        NONE: Plain generation — scheduler is disabled, 100% original HF behavior.
              This is the default mode.
        INTERNAL: LLM-driven scheduling — the model itself generates control tokens
                  (e.g. ``<STATE:READ_NEXT_CHUNK>``, ``<RECALL:chunk_3>``) that the
                  scheduler intercepts and executes. The model drives the state machine.
        FORCE: External scheduling — external code fully controls the generation
               pipeline via the ``GenerationScheduler`` API (pause, resume, inject,
               force_token). Useful for deterministic pipelines.
    """

    NONE = "none"
    INTERNAL = "internal"
    FORCE = "force"


class GenerationPhase(Enum):
    """
    Explicit phases of the transformers `generate()` lifecycle.

    Corresponds to the 5-stage decomposition proposed in Issue #30810,
    extended with CHECKING, INJECTING, and ERROR phases for full
    scheduler support.

    Phase transitions:
        IDLE → INIT → PREFILL → DECODING ⇄ CHECKING ⇄ INJECTING → POSTPROCESS → COMPLETE
                                                                          ↓
                                                                        ERROR (from any phase)
    """

    IDLE = auto()  # Not started
    INIT = auto()  # Parameter validation, input preparation
    PREFILL = auto()  # First forward pass, filling KV Cache
    DECODING = auto()  # Main decode loop (token-by-token)
    CHECKING = auto()  # User-defined checkpoint (scheduler pauses here)
    INJECTING = auto()  # External token / KV / context injection
    POSTPROCESS = auto()  # Post-processing (beam selection, length penalty, etc.)
    COMPLETE = auto()  # Generation finished
    ERROR = auto()  # Error state


# Valid phase transitions (from_phase -> set of allowed to_phases)
VALID_TRANSITIONS: dict[GenerationPhase, set[GenerationPhase]] = {
    GenerationPhase.IDLE: {GenerationPhase.INIT, GenerationPhase.ERROR},
    GenerationPhase.INIT: {GenerationPhase.PREFILL, GenerationPhase.ERROR},
    GenerationPhase.PREFILL: {GenerationPhase.DECODING, GenerationPhase.ERROR},
    GenerationPhase.DECODING: {
        GenerationPhase.DECODING,  # next token step
        GenerationPhase.CHECKING,
        GenerationPhase.INJECTING,
        GenerationPhase.POSTPROCESS,
        GenerationPhase.ERROR,
    },
    GenerationPhase.CHECKING: {
        GenerationPhase.DECODING,
        GenerationPhase.INJECTING,
        GenerationPhase.POSTPROCESS,
        GenerationPhase.ERROR,
    },
    GenerationPhase.INJECTING: {
        GenerationPhase.DECODING,
        GenerationPhase.PREFILL,  # re-prefill after injection
        GenerationPhase.ERROR,
    },
    GenerationPhase.POSTPROCESS: {GenerationPhase.COMPLETE, GenerationPhase.ERROR},
    GenerationPhase.COMPLETE: set(),  # Terminal state
    GenerationPhase.ERROR: set(),  # Terminal state
}


@dataclass
class GenerationState:
    """
    A serializable snapshot of the full generation state at any point.

    Design principles:
        - Contains ALL information needed to reproduce the current generation progress.
        - Serializable via `save()` / `load()` for pause/resume/checkpoint.
        - Lightweight: does not copy large tensors by default (uses references + dirty flags).
        - Extensible via `metadata` dict for user-defined data.

    Attributes:
        phase (`GenerationPhase`):
            Current generation phase.
        step (`int`):
            Current decoding step number (0-indexed).
        input_ids (`torch.LongTensor`):
            All token IDs generated so far, shape `(batch_size, seq_len)`.
        next_token_logits (`torch.FloatTensor`, *optional*):
            Raw logits from the latest forward pass, shape `(batch_size, vocab_size)`.
        next_token_scores (`torch.FloatTensor`, *optional*):
            Processed scores (after LogitsProcessor), shape `(batch_size, vocab_size)`.
        next_tokens (`torch.LongTensor`, *optional*):
            The token(s) selected at the current step, shape `(batch_size,)`.
        past_key_values (*optional*):
            Reference to the KV Cache object.
        attention_mask (`torch.LongTensor`, *optional*):
            Current attention mask.
        position_ids (`torch.LongTensor`, *optional*):
            Current position IDs.
        stopping_criteria_met (`bool`):
            Whether any stopping criterion has been triggered.
        eos_token_generated (`bool`):
            Whether an EOS token has been generated for all sequences.
        unfinished_sequences (`torch.LongTensor`, *optional*):
            Mask of sequences still being generated, shape `(batch_size,)`.
        model_kwargs (`Dict[str, Any]`):
            Additional model keyword arguments (encoder_outputs, etc.).
        metadata (`Dict[str, Any]`):
            User-defined extension data.
        timestamp (`float`):
            Wall-clock time when this state was created.
    """

    phase: GenerationPhase = GenerationPhase.IDLE
    step: int = 0
    input_ids: torch.LongTensor | None = None
    next_token_logits: torch.FloatTensor | None = None
    next_token_scores: torch.FloatTensor | None = None
    next_tokens: torch.LongTensor | None = None
    past_key_values: Any | None = None
    attention_mask: torch.LongTensor | None = None
    position_ids: torch.LongTensor | None = None

    # Status flags
    stopping_criteria_met: bool = False
    eos_token_generated: bool = False
    unfinished_sequences: torch.LongTensor | None = None

    # Batch-level scheduler control mask: shape (batch_size,), 1 = active, 0 = paused
    # Used by FORCE mode to pause individual sequences in a batch.
    batch_control_mask: torch.LongTensor | None = None

    # Model kwargs (for encoder-decoder, etc.)
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # User-defined extension
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: float = field(default_factory=time.time)

    def clone(self, deep_copy_tensors: bool = False) -> "GenerationState":
        """
        Create a copy of this state.

        Args:
            deep_copy_tensors: If True, deep-copy all tensors (expensive but safe).
                              If False, only copy metadata and scalar fields.

        Returns:
            A new `GenerationState` with copied data.
        """
        if deep_copy_tensors:
            return copy.deepcopy(self)

        # Shallow copy: reference tensors, deep-copy dicts
        return GenerationState(
            phase=self.phase,
            step=self.step,
            input_ids=self.input_ids,
            next_token_logits=self.next_token_logits,
            next_token_scores=self.next_token_scores,
            next_tokens=self.next_tokens,
            past_key_values=self.past_key_values,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            stopping_criteria_met=self.stopping_criteria_met,
            eos_token_generated=self.eos_token_generated,
            unfinished_sequences=self.unfinished_sequences,
            batch_control_mask=self.batch_control_mask,
            model_kwargs=copy.copy(self.model_kwargs),
            metadata=copy.deepcopy(self.metadata),
            timestamp=time.time(),
        )

    def save(self, path: str):
        """
        Serialize state to disk (excluding non-serializable references like KV Cache).

        Args:
            path: File path to save the state checkpoint.
        """
        checkpoint = {
            "phase": self.phase.name,
            "step": self.step,
            "stopping_criteria_met": self.stopping_criteria_met,
            "eos_token_generated": self.eos_token_generated,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        # Save tensors that are present
        tensor_fields = [
            "input_ids", "next_token_logits", "next_token_scores",
            "next_tokens", "attention_mask", "position_ids", "unfinished_sequences",
            "batch_control_mask",
        ]
        for field_name in tensor_fields:
            value = getattr(self, field_name)
            if value is not None:
                checkpoint[field_name] = value.cpu()

        torch.save(checkpoint, path)
        logger.info(f"Generation state saved to {path} (phase={self.phase.name}, step={self.step})")

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "GenerationState":
        """
        Load state from disk.

        Args:
            path: File path to load the state checkpoint from.
            device: Device to move tensors to. If None, keeps on CPU.

        Returns:
            A restored `GenerationState` (without KV Cache — must be rebuilt).
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        state = cls(
            phase=GenerationPhase[checkpoint["phase"]],
            step=checkpoint["step"],
            stopping_criteria_met=checkpoint["stopping_criteria_met"],
            eos_token_generated=checkpoint["eos_token_generated"],
            metadata=checkpoint.get("metadata", {}),
            timestamp=checkpoint.get("timestamp", time.time()),
        )

        tensor_fields = [
            "input_ids", "next_token_logits", "next_token_scores",
            "next_tokens", "attention_mask", "position_ids", "unfinished_sequences",
            "batch_control_mask",
        ]
        for field_name in tensor_fields:
            if field_name in checkpoint:
                value = checkpoint[field_name]
                if device is not None:
                    value = value.to(device)
                setattr(state, field_name, value)

        logger.info(f"Generation state loaded from {path} (phase={state.phase.name}, step={state.step})")
        return state


class GenerationStateMachine:
    """
    Manages phase transitions for the generation lifecycle.

    Validates transitions against the allowed transition graph and
    maintains a history of phase changes for debugging/observability.

    Usage:
        ```python
        sm = GenerationStateMachine()
        sm.transition_to(GenerationPhase.INIT)
        sm.transition_to(GenerationPhase.PREFILL)
        sm.transition_to(GenerationPhase.DECODING)
        # ...
        sm.transition_to(GenerationPhase.COMPLETE)
        ```
    """

    def __init__(self):
        self._phase: GenerationPhase = GenerationPhase.IDLE
        self._state: GenerationState = GenerationState(phase=GenerationPhase.IDLE)
        self._history: list[tuple[GenerationPhase, GenerationPhase, float]] = []
        self._transition_count: int = 0

    @property
    def phase(self) -> GenerationPhase:
        """Current generation phase."""
        return self._phase

    @property
    def current_state(self) -> GenerationState:
        """Current generation state snapshot."""
        return self._state

    @current_state.setter
    def current_state(self, state: GenerationState):
        """Set the current state (used by scheduler to update state)."""
        self._state = state
        self._phase = state.phase

    @property
    def history(self) -> list[tuple[GenerationPhase, GenerationPhase, float]]:
        """List of (from_phase, to_phase, timestamp) transitions."""
        return self._history

    @property
    def transition_count(self) -> int:
        """Total number of phase transitions."""
        return self._transition_count

    def transition_to(self, new_phase: GenerationPhase, validate: bool = True) -> GenerationPhase:
        """
        Transition to a new phase.

        Args:
            new_phase: The target phase.
            validate: Whether to validate the transition against VALID_TRANSITIONS.

        Returns:
            The previous phase.

        Raises:
            ValueError: If the transition is invalid and `validate=True`.
        """
        old_phase = self._phase

        if validate and new_phase not in VALID_TRANSITIONS.get(old_phase, set()):
            raise ValueError(
                f"Invalid generation phase transition: {old_phase.name} → {new_phase.name}. "
                f"Allowed transitions from {old_phase.name}: "
                f"{[p.name for p in VALID_TRANSITIONS.get(old_phase, set())]}"
            )

        self._phase = new_phase
        self._state.phase = new_phase
        self._history.append((old_phase, new_phase, time.time()))
        self._transition_count += 1

        logger.debug(f"Generation phase transition: {old_phase.name} → {new_phase.name} (step={self._state.step})")

        return old_phase

    def is_terminal(self) -> bool:
        """Whether the current phase is a terminal state (COMPLETE or ERROR)."""
        return self._phase in (GenerationPhase.COMPLETE, GenerationPhase.ERROR)

    def is_decoding(self) -> bool:
        """Whether we are in the main decoding loop."""
        return self._phase == GenerationPhase.DECODING

    def reset(self):
        """Reset the state machine to IDLE."""
        self._phase = GenerationPhase.IDLE
        self._state = GenerationState(phase=GenerationPhase.IDLE)
        self._history.clear()
        self._transition_count = 0

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the state machine for debugging.

        Returns:
            Dict with phase, step, transition count, and history.
        """
        return {
            "current_phase": self._phase.name,
            "step": self._state.step,
            "transition_count": self._transition_count,
            "history": [(f.name, t.name, ts) for f, t, ts in self._history[-20:]],  # last 20
            "is_terminal": self.is_terminal(),
        }
