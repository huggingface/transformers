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

"""Tests for the Generation State Machine module, including SchedulerMode."""

import os
import tempfile
import unittest

import torch

from transformers.generation.state_machine import (
    VALID_TRANSITIONS,
    GenerationPhase,
    GenerationState,
    GenerationStateMachine,
    SchedulerMode,
)


class TestSchedulerMode(unittest.TestCase):
    """Tests for the SchedulerMode enum."""

    def test_all_modes_exist(self):
        """All expected modes are defined."""
        expected = ["NONE", "INTERNAL", "FORCE"]
        actual = [m.name for m in SchedulerMode]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_mode_values(self):
        """Mode values are lowercase strings."""
        self.assertEqual(SchedulerMode.NONE.value, "none")
        self.assertEqual(SchedulerMode.INTERNAL.value, "internal")
        self.assertEqual(SchedulerMode.FORCE.value, "force")

    def test_mode_from_string(self):
        """Modes can be created from string values."""
        self.assertEqual(SchedulerMode("none"), SchedulerMode.NONE)
        self.assertEqual(SchedulerMode("internal"), SchedulerMode.INTERNAL)
        self.assertEqual(SchedulerMode("force"), SchedulerMode.FORCE)

    def test_invalid_mode_raises(self):
        """Invalid mode string raises ValueError."""
        with self.assertRaises(ValueError):
            SchedulerMode("invalid_mode")


class TestGenerationPhase(unittest.TestCase):
    """Tests for the GenerationPhase enum."""

    def test_all_phases_exist(self):
        """All expected phases are defined."""
        expected = [
            "IDLE", "INIT", "PREFILL", "DECODING", "CHECKING",
            "INJECTING", "POSTPROCESS", "COMPLETE", "ERROR",
        ]
        actual = [p.name for p in GenerationPhase]
        self.assertEqual(sorted(actual), sorted(expected))

    def test_phases_are_unique(self):
        """All phase values are unique."""
        values = [p.value for p in GenerationPhase]
        self.assertEqual(len(values), len(set(values)))

    def test_valid_transitions_covers_all_phases(self):
        """VALID_TRANSITIONS has entries for all phases."""
        for phase in GenerationPhase:
            self.assertIn(phase, VALID_TRANSITIONS)

    def test_terminal_states_have_no_transitions(self):
        """COMPLETE and ERROR are terminal (no outgoing transitions)."""
        self.assertEqual(VALID_TRANSITIONS[GenerationPhase.COMPLETE], set())
        self.assertEqual(VALID_TRANSITIONS[GenerationPhase.ERROR], set())

    def test_error_reachable_from_all_non_terminal(self):
        """ERROR is reachable from all non-terminal phases."""
        for phase in GenerationPhase:
            if phase not in (GenerationPhase.COMPLETE, GenerationPhase.ERROR):
                self.assertIn(
                    GenerationPhase.ERROR,
                    VALID_TRANSITIONS[phase],
                    f"ERROR should be reachable from {phase.name}",
                )

    def test_happy_path_transitions_valid(self):
        """The normal generation path is valid."""
        happy_path = [
            GenerationPhase.IDLE,
            GenerationPhase.INIT,
            GenerationPhase.PREFILL,
            GenerationPhase.DECODING,
            GenerationPhase.POSTPROCESS,
            GenerationPhase.COMPLETE,
        ]
        for i in range(len(happy_path) - 1):
            from_phase = happy_path[i]
            to_phase = happy_path[i + 1]
            self.assertIn(
                to_phase,
                VALID_TRANSITIONS[from_phase],
                f"Transition {from_phase.name} → {to_phase.name} should be valid",
            )


class TestGenerationState(unittest.TestCase):
    """Tests for the GenerationState dataclass."""

    def test_default_state(self):
        """Default state is IDLE with step 0."""
        state = GenerationState()
        self.assertEqual(state.phase, GenerationPhase.IDLE)
        self.assertEqual(state.step, 0)
        self.assertIsNone(state.input_ids)
        self.assertFalse(state.stopping_criteria_met)
        self.assertFalse(state.eos_token_generated)
        self.assertIsInstance(state.metadata, dict)
        self.assertIsInstance(state.model_kwargs, dict)

    def test_state_with_tensors(self):
        """State can hold tensor data."""
        input_ids = torch.randint(0, 1000, (2, 10))
        logits = torch.randn(2, 50000)
        unfinished = torch.ones(2, dtype=torch.long)

        state = GenerationState(
            phase=GenerationPhase.DECODING,
            step=5,
            input_ids=input_ids,
            next_token_logits=logits,
            unfinished_sequences=unfinished,
        )

        self.assertEqual(state.phase, GenerationPhase.DECODING)
        self.assertEqual(state.step, 5)
        self.assertTrue(torch.equal(state.input_ids, input_ids))
        self.assertTrue(torch.equal(state.next_token_logits, logits))
        self.assertTrue(torch.equal(state.unfinished_sequences, unfinished))

    def test_clone_shallow(self):
        """Shallow clone shares tensor references but copies metadata."""
        state = GenerationState(
            phase=GenerationPhase.DECODING,
            step=3,
            input_ids=torch.randint(0, 100, (1, 5)),
            metadata={"key": "value"},
        )
        clone = state.clone(deep_copy_tensors=False)

        # Scalar fields are copied
        self.assertEqual(clone.phase, state.phase)
        self.assertEqual(clone.step, state.step)

        # Tensors are shared (same object)
        self.assertIs(clone.input_ids, state.input_ids)

        # Metadata is deep-copied
        clone.metadata["key"] = "modified"
        self.assertEqual(state.metadata["key"], "value")

    def test_clone_deep(self):
        """Deep clone copies all tensors."""
        state = GenerationState(
            phase=GenerationPhase.DECODING,
            step=3,
            input_ids=torch.randint(0, 100, (1, 5)),
        )
        clone = state.clone(deep_copy_tensors=True)

        # Tensors are different objects
        self.assertIsNot(clone.input_ids, state.input_ids)
        # But have the same values
        self.assertTrue(torch.equal(clone.input_ids, state.input_ids))

    def test_save_and_load(self):
        """State can be serialized and deserialized."""
        state = GenerationState(
            phase=GenerationPhase.DECODING,
            step=10,
            input_ids=torch.randint(0, 100, (2, 15)),
            next_token_logits=torch.randn(2, 1000),
            attention_mask=torch.ones(2, 15, dtype=torch.long),
            unfinished_sequences=torch.ones(2, dtype=torch.long),
            stopping_criteria_met=False,
            eos_token_generated=False,
            metadata={"test_key": "test_value", "step_count": 10},
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            state.save(path)
            loaded = GenerationState.load(path)

            self.assertEqual(loaded.phase, GenerationPhase.DECODING)
            self.assertEqual(loaded.step, 10)
            self.assertTrue(torch.equal(loaded.input_ids, state.input_ids.cpu()))
            self.assertTrue(torch.equal(loaded.next_token_logits, state.next_token_logits.cpu()))
            self.assertTrue(torch.equal(loaded.attention_mask, state.attention_mask.cpu()))
            self.assertEqual(loaded.metadata["test_key"], "test_value")
            self.assertFalse(loaded.stopping_criteria_met)
        finally:
            os.unlink(path)

    def test_save_load_minimal_state(self):
        """Minimal state (no tensors) can be saved and loaded."""
        state = GenerationState(
            phase=GenerationPhase.IDLE,
            step=0,
            metadata={"info": "test"},
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            state.save(path)
            loaded = GenerationState.load(path)
            self.assertEqual(loaded.phase, GenerationPhase.IDLE)
            self.assertEqual(loaded.step, 0)
            self.assertIsNone(loaded.input_ids)
        finally:
            os.unlink(path)


class TestGenerationStateMachine(unittest.TestCase):
    """Tests for the GenerationStateMachine."""

    def setUp(self):
        self.sm = GenerationStateMachine()

    def test_initial_state(self):
        """State machine starts in IDLE phase."""
        self.assertEqual(self.sm.phase, GenerationPhase.IDLE)
        self.assertEqual(self.sm.current_state.phase, GenerationPhase.IDLE)
        self.assertEqual(self.sm.transition_count, 0)
        self.assertEqual(len(self.sm.history), 0)

    def test_valid_transition(self):
        """Valid transitions succeed and update state."""
        old = self.sm.transition_to(GenerationPhase.INIT)
        self.assertEqual(old, GenerationPhase.IDLE)
        self.assertEqual(self.sm.phase, GenerationPhase.INIT)
        self.assertEqual(self.sm.transition_count, 1)
        self.assertEqual(len(self.sm.history), 1)

    def test_invalid_transition_raises(self):
        """Invalid transitions raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.sm.transition_to(GenerationPhase.DECODING)  # IDLE → DECODING is invalid
        self.assertIn("Invalid generation phase transition", str(ctx.exception))

    def test_invalid_transition_no_validate(self):
        """Invalid transitions succeed when validate=False."""
        self.sm.transition_to(GenerationPhase.DECODING, validate=False)
        self.assertEqual(self.sm.phase, GenerationPhase.DECODING)

    def test_full_happy_path(self):
        """Complete happy path through all phases."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)
        # Simulate multiple decode steps
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.POSTPROCESS)
        self.sm.transition_to(GenerationPhase.COMPLETE)

        self.assertTrue(self.sm.is_terminal())
        self.assertEqual(self.sm.transition_count, 7)

    def test_checking_phase(self):
        """DECODING → CHECKING → DECODING transition."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.CHECKING)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.assertEqual(self.sm.phase, GenerationPhase.DECODING)

    def test_injection_phase(self):
        """DECODING → INJECTING → DECODING transition."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.INJECTING)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.assertEqual(self.sm.phase, GenerationPhase.DECODING)

    def test_injection_to_prefill(self):
        """INJECTING → PREFILL (re-prefill after injection)."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.INJECTING)
        self.sm.transition_to(GenerationPhase.PREFILL)  # re-prefill
        self.assertEqual(self.sm.phase, GenerationPhase.PREFILL)

    def test_error_from_any_phase(self):
        """ERROR is reachable from any non-terminal phase."""
        for phase in GenerationPhase:
            if phase in (GenerationPhase.COMPLETE, GenerationPhase.ERROR):
                continue
            sm = GenerationStateMachine()
            sm._phase = phase  # Force to this phase
            sm.transition_to(GenerationPhase.ERROR)
            self.assertTrue(sm.is_terminal())

    def test_terminal_states(self):
        """COMPLETE and ERROR are terminal."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)
        self.sm.transition_to(GenerationPhase.POSTPROCESS)
        self.sm.transition_to(GenerationPhase.COMPLETE)
        self.assertTrue(self.sm.is_terminal())

        # Cannot transition from COMPLETE
        with self.assertRaises(ValueError):
            self.sm.transition_to(GenerationPhase.DECODING)

    def test_is_decoding(self):
        """is_decoding() returns True only in DECODING phase."""
        self.assertFalse(self.sm.is_decoding())
        self.sm.transition_to(GenerationPhase.INIT)
        self.assertFalse(self.sm.is_decoding())
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.assertFalse(self.sm.is_decoding())
        self.sm.transition_to(GenerationPhase.DECODING)
        self.assertTrue(self.sm.is_decoding())

    def test_reset(self):
        """Reset returns to IDLE with clean state."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.reset()

        self.assertEqual(self.sm.phase, GenerationPhase.IDLE)
        self.assertEqual(self.sm.transition_count, 0)
        self.assertEqual(len(self.sm.history), 0)

    def test_history_tracking(self):
        """History correctly records transitions."""
        self.sm.transition_to(GenerationPhase.INIT)
        self.sm.transition_to(GenerationPhase.PREFILL)
        self.sm.transition_to(GenerationPhase.DECODING)

        history = self.sm.history
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0][0], GenerationPhase.IDLE)
        self.assertEqual(history[0][1], GenerationPhase.INIT)
        self.assertEqual(history[1][0], GenerationPhase.INIT)
        self.assertEqual(history[1][1], GenerationPhase.PREFILL)
        self.assertEqual(history[2][0], GenerationPhase.PREFILL)
        self.assertEqual(history[2][1], GenerationPhase.DECODING)

    def test_get_summary(self):
        """Summary contains expected keys."""
        self.sm.transition_to(GenerationPhase.INIT)
        summary = self.sm.get_summary()

        self.assertIn("current_phase", summary)
        self.assertIn("step", summary)
        self.assertIn("transition_count", summary)
        self.assertIn("history", summary)
        self.assertIn("is_terminal", summary)
        self.assertEqual(summary["current_phase"], "INIT")
        self.assertFalse(summary["is_terminal"])

    def test_current_state_setter(self):
        """Setting current_state updates both state and phase."""
        new_state = GenerationState(phase=GenerationPhase.DECODING, step=5)
        self.sm.current_state = new_state
        self.assertEqual(self.sm.phase, GenerationPhase.DECODING)
        self.assertEqual(self.sm.current_state.step, 5)


if __name__ == "__main__":
    unittest.main()
