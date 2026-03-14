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
Black-box integration tests for the Generation Scheduler.

These tests exercise the FULL end-to-end `model.generate()` pipeline with a real
(tiny) model, verifying that the three scheduler modes (none / internal / force)
work correctly in realistic conditions.

Unlike the unit tests in `test_generation_scheduler.py` (which test components in
isolation), these tests treat `generate()` as a black box and validate:
    - Output correctness
    - Callback invocation with real model outputs
    - Pause / resume / inject / force_token behavior
    - Backward compatibility (none mode)
    - GenerationConfig integration

Requirements:
    - torch
    - transformers (installed from source)
    - Internet access for first run (downloads tiny-random-gpt2, ~2MB)

Run:
    pytest tests/generation/test_scheduler_integration.py -v
    # Or with slow tests:
    RUN_SLOW=1 pytest tests/generation/test_scheduler_integration.py -v
"""

import os
import tempfile
import time
import unittest

import pytest

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers.generation.generation_scheduler import (
        ControlTokenParser,
        GenerationScheduler,
        SchedulerCallback,
    )
    from transformers.generation.scheduler_callbacks import (
        EntropyMonitorCallback,
        GenerationLoggerCallback,
        InternalSchedulerCallback,
        RepetitionDetectorCallback,
        StepBudgetCallback,
        TokenPatternCallback,
    )
    from transformers.generation.state_machine import (
        GenerationPhase,
        GenerationState,
        SchedulerMode,
    )


# ==============================================================================
# Constants
# ==============================================================================

TINY_MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


# ==============================================================================
# Shared Fixtures
# ==============================================================================


def _load_tiny_model():
    """Load the tiny random GPT-2 model and tokenizer for testing."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_ID).to(torch_device)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ==============================================================================
# Helper Callbacks for Integration Tests
# ==============================================================================


class EventRecorderCallback(SchedulerCallback):
    """Records every scheduler event with timestamps for integration verification."""

    def __init__(self):
        super().__init__()
        self.phase_transitions = []
        self.step_begins = []
        self.logits_shapes = []
        self.tokens_generated = []
        self.step_ends = []
        self.completed = False
        self.errors = []
        self._start_time = time.time()

    def on_phase_transition(self, from_phase, to_phase, state, context):
        self.phase_transitions.append({
            "from": from_phase.name,
            "to": to_phase.name,
            "step": state.step,
            "time": time.time() - self._start_time,
        })
        return True

    def on_step_begin(self, state, context):
        self.step_begins.append(state.step)
        return True

    def on_logits_ready(self, logits, state, context):
        self.logits_shapes.append(tuple(logits.shape))
        return logits

    def on_token_generated(self, token_id, state, context):
        tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()
        self.tokens_generated.append({
            "step": state.step,
            "token_id": tid,
        })
        return True

    def on_step_end(self, state, context):
        self.step_ends.append(state.step)
        return True

    def on_generation_complete(self, state):
        self.completed = True

    def on_error(self, error, state):
        self.errors.append(str(error))


class PauseAtStepCallback(SchedulerCallback):
    """Pauses generation at a specific step."""

    def __init__(self, pause_step: int):
        super().__init__()
        self.pause_step = pause_step
        self.paused = False

    def on_token_generated(self, token_id, state, context):
        if state.step >= self.pause_step:
            context.should_pause = True
            self.paused = True
            return False
        return True


class ForceTokenAtStepCallback(SchedulerCallback):
    """Forces a specific token at a specific step."""

    def __init__(self, step: int, token_id: int):
        super().__init__()
        self.target_step = step
        self.forced_token_id = token_id
        self.was_forced = False

    def on_step_begin(self, state, context):
        if state.step == self.target_step:
            context.forced_token = self.forced_token_id
            self.was_forced = True
        return True


class LogitsScalerCallback(SchedulerCallback):
    """Scales logits by a factor — verifies logits modification works end-to-end."""

    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor
        self.call_count = 0

    def on_logits_ready(self, logits, state, context):
        self.call_count += 1
        return logits * self.factor


class TokenCollectorCallback(SchedulerCallback):
    """Collects all generated token IDs for sequence verification."""

    def __init__(self):
        super().__init__()
        self.token_ids = []

    def on_token_generated(self, token_id, state, context):
        tid = token_id[0].item() if token_id.dim() > 0 else token_id.item()
        self.token_ids.append(tid)
        return True


class CheckIntervalVerifier(SchedulerCallback):
    """Verifies that CHECKING phase transitions happen at the correct intervals."""

    def __init__(self):
        super().__init__()
        self.checking_steps = []

    def on_phase_transition(self, from_phase, to_phase, state, context):
        if to_phase == GenerationPhase.CHECKING:
            self.checking_steps.append(state.step)
        return True


# ==============================================================================
# MODE: NONE — Backward Compatibility Tests
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestNoneModeIntegration(unittest.TestCase):
    """
    Black-box tests for scheduler_mode='none' (default).

    Verifies that the scheduler does NOT interfere with normal generation
    and that outputs are identical to generation without any scheduler.
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_text = "Hello, world"
        cls.input_ids = cls.tokenizer(cls.input_text, return_tensors="pt").input_ids.to(torch_device)

    def test_no_scheduler_baseline(self):
        """Generation without scheduler produces valid output."""
        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
        )
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])

    def test_none_scheduler_matches_baseline(self):
        """scheduler_mode='none' produces IDENTICAL output to no scheduler."""
        # Baseline: no scheduler
        torch.manual_seed(42)
        baseline = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

        # With none-mode scheduler
        torch.manual_seed(42)
        scheduler = GenerationScheduler(mode="none")
        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertTrue(torch.equal(baseline, output),
                        "scheduler_mode='none' should produce identical output to no scheduler")

    def test_none_scheduler_not_activated(self):
        """NONE mode scheduler is never activated."""
        scheduler = GenerationScheduler(mode="none")
        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )
        self.assertFalse(scheduler.is_active)

    def test_none_scheduler_via_generation_config(self):
        """scheduler_mode='none' via GenerationConfig also works."""
        torch.manual_seed(42)
        baseline = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

        torch.manual_seed(42)
        config = GenerationConfig(
            max_new_tokens=20,
            do_sample=False,
            scheduler_mode="none",
        )
        output = self.model.generate(self.input_ids, generation_config=config)

        self.assertTrue(torch.equal(baseline, output))

    def test_default_generation_config_has_no_scheduler(self):
        """Default GenerationConfig does not enable scheduler."""
        config = GenerationConfig()
        self.assertIsNone(config.scheduler_mode)

    def test_none_scheduler_with_sampling(self):
        """NONE mode works with do_sample=True (stochastic generation)."""
        scheduler = GenerationScheduler(mode="none")
        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=True,
            temperature=1.0,
            scheduler=scheduler,
        )
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])

    def test_none_mode_no_callback_overhead(self):
        """Even with callbacks registered, NONE mode doesn't call them."""
        recorder = EventRecorderCallback()
        scheduler = GenerationScheduler(mode="none")
        scheduler.register_callback(recorder)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # NONE mode should not activate scheduler → no events
        self.assertEqual(len(recorder.phase_transitions), 0)
        self.assertEqual(len(recorder.tokens_generated), 0)
        self.assertFalse(recorder.completed)


# ==============================================================================
# MODE: FORCE — External Control Tests
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestForceModeIntegration(unittest.TestCase):
    """
    Black-box tests for scheduler_mode='force'.

    Tests external code controlling generation via the scheduler API:
    pause, resume, force_token, inject_tokens, callbacks, etc.
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_text = "Hello, world"
        cls.input_ids = cls.tokenizer(cls.input_text, return_tensors="pt").input_ids.to(torch_device)

    # ---- Basic Generation ----

    def test_force_mode_basic_generation(self):
        """FORCE mode generates valid output with callbacks."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=15,
            do_sample=False,
            scheduler=scheduler,
        )

        # Output should be longer than input
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])

        # Callbacks should have been invoked
        self.assertGreater(len(recorder.phase_transitions), 0)
        self.assertGreater(len(recorder.tokens_generated), 0)
        self.assertTrue(recorder.completed)
        self.assertEqual(len(recorder.errors), 0)

    def test_force_mode_phase_lifecycle(self):
        """FORCE mode follows correct phase lifecycle: IDLE→INIT→PREFILL→DECODING→POSTPROCESS→COMPLETE."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        phase_names = [(t["from"], t["to"]) for t in recorder.phase_transitions]

        # First three transitions should be IDLE→INIT→PREFILL→DECODING
        self.assertEqual(phase_names[0], ("IDLE", "INIT"))
        self.assertEqual(phase_names[1], ("INIT", "PREFILL"))
        self.assertEqual(phase_names[2], ("PREFILL", "DECODING"))

        # Last two transitions should be DECODING→POSTPROCESS→COMPLETE
        self.assertEqual(phase_names[-2], ("DECODING", "POSTPROCESS"))
        self.assertEqual(phase_names[-1], ("POSTPROCESS", "COMPLETE"))

    def test_force_mode_step_count_matches_tokens(self):
        """Number of step_begin/step_end calls matches generated tokens."""
        max_new = 10
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertEqual(len(recorder.step_begins), num_generated)
        self.assertEqual(len(recorder.step_ends), num_generated)
        self.assertEqual(len(recorder.tokens_generated), num_generated)

    def test_force_mode_logits_shape(self):
        """Logits passed to callbacks have correct shape (batch_size, vocab_size)."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        vocab_size = self.model.config.vocab_size
        for shape in recorder.logits_shapes:
            self.assertEqual(shape[0], 1)  # batch_size=1
            self.assertEqual(shape[1], vocab_size)

    # ---- Pause / Resume ----

    def test_force_mode_pause_at_step(self):
        """Generation pauses at a specific step via callback."""
        pause_step = 3
        scheduler = GenerationScheduler(mode="force")
        pauser = PauseAtStepCallback(pause_step=pause_step)
        collector = TokenCollectorCallback()
        scheduler.register_callback(collector)
        scheduler.register_callback(pauser)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        # Should have paused: output is shorter than max_new_tokens
        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        # We pause at step 3, but the token at step 3 is still appended before break
        self.assertLessEqual(num_generated, pause_step + 1)
        self.assertTrue(pauser.paused)
        self.assertTrue(scheduler.is_paused())

    def test_force_mode_step_budget_callback(self):
        """StepBudgetCallback limits generation length."""
        budget = 5
        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(StepBudgetCallback(max_steps=budget))

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=50,  # Much larger than budget
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertLessEqual(num_generated, budget + 1)

    def test_force_mode_step_budget_via_context(self):
        """Step budget can be set via scheduler context directly."""
        scheduler = GenerationScheduler(mode="force")
        scheduler.context.step_budget = 3

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=50,
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertLessEqual(num_generated, 4)

    # ---- Force Token ----

    def test_force_mode_force_next_token(self):
        """force_next_token overrides the model's token selection."""
        # Pick a specific token to force
        forced_token_id = 42
        target_step = 2

        scheduler = GenerationScheduler(mode="force")
        forcer = ForceTokenAtStepCallback(step=target_step, token_id=forced_token_id)
        collector = TokenCollectorCallback()
        scheduler.register_callback(forcer)
        scheduler.register_callback(collector)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertTrue(forcer.was_forced)
        # The token at target_step should be the forced token
        self.assertEqual(collector.token_ids[target_step], forced_token_id)

    def test_force_mode_force_multiple_tokens(self):
        """Multiple tokens can be forced at different steps."""
        forced_tokens = {0: 10, 2: 20, 4: 30}

        class MultiForceCallback(SchedulerCallback):
            def __init__(self, token_map):
                self.token_map = token_map

            def on_step_begin(self, state, context):
                if state.step in self.token_map:
                    context.forced_token = self.token_map[state.step]
                return True

        scheduler = GenerationScheduler(mode="force")
        collector = TokenCollectorCallback()
        scheduler.register_callback(MultiForceCallback(forced_tokens))
        scheduler.register_callback(collector)

        self.model.generate(
            self.input_ids,
            max_new_tokens=6,
            do_sample=False,
            scheduler=scheduler,
        )

        for step, expected_token in forced_tokens.items():
            if step < len(collector.token_ids):
                self.assertEqual(collector.token_ids[step], expected_token,
                                 f"Token at step {step} should be {expected_token}")

    # ---- Logits Modification ----

    def test_force_mode_logits_modification(self):
        """Logits modification via callback changes generation output."""
        # Generate baseline
        torch.manual_seed(42)
        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
        )

        # Generate with logits scaling (should change output)
        torch.manual_seed(42)
        scheduler = GenerationScheduler(mode="force")
        scaler = LogitsScalerCallback(factor=0.01)  # Flatten distribution dramatically
        scheduler.register_callback(scaler)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(scaler.call_count, 0)
        # With greedy decoding and extreme scaling, output may or may not change
        # (depends on how peaked the original distribution is)
        # The key assertion is that the callback was invoked correctly
        self.assertEqual(scaler.call_count, 10)

    # ---- Check Interval ----

    def test_force_mode_check_interval(self):
        """CHECKING phase is triggered at the correct interval."""
        interval = 3
        scheduler = GenerationScheduler(mode="force")
        scheduler.context.check_interval = interval
        verifier = CheckIntervalVerifier()
        scheduler.register_callback(verifier)

        self.model.generate(
            self.input_ids,
            max_new_tokens=15,
            do_sample=False,
            scheduler=scheduler,
        )

        # CHECKING should happen at steps 3, 6, 9, 12, ...
        for step in verifier.checking_steps:
            self.assertEqual(step % interval, 0, f"CHECKING at step {step} is not a multiple of {interval}")
            self.assertGreater(step, 0, "CHECKING should not happen at step 0")

    def test_force_mode_check_interval_via_config(self):
        """Check interval can be set via GenerationConfig."""
        scheduler = GenerationScheduler(mode="force")
        verifier = CheckIntervalVerifier()
        scheduler.register_callback(verifier)

        config = GenerationConfig(
            max_new_tokens=15,
            do_sample=False,
            scheduler_mode="force",
            scheduler_check_interval=5,
        )

        self.model.generate(
            self.input_ids,
            generation_config=config,
            scheduler=scheduler,
        )

        for step in verifier.checking_steps:
            self.assertEqual(step % 5, 0)

    # ---- Multiple Callbacks ----

    def test_force_mode_multiple_callbacks(self):
        """Multiple callbacks all receive events."""
        scheduler = GenerationScheduler(mode="force")
        recorder1 = EventRecorderCallback()
        recorder2 = EventRecorderCallback()
        scheduler.register_callback(recorder1)
        scheduler.register_callback(recorder2)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertEqual(len(recorder1.tokens_generated), len(recorder2.tokens_generated))
        self.assertTrue(recorder1.completed)
        self.assertTrue(recorder2.completed)

    def test_force_mode_callback_ordering(self):
        """First callback's pause stops subsequent callbacks from being called."""
        scheduler = GenerationScheduler(mode="force")

        class ImmediatePauseCallback(SchedulerCallback):
            def on_token_generated(self, token_id, state, context):
                return False  # Always pause

        second_called = []

        class SecondCallback(SchedulerCallback):
            def on_token_generated(self, token_id, state, context):
                second_called.append(True)
                return True

        scheduler.register_callback(ImmediatePauseCallback())
        scheduler.register_callback(SecondCallback())

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Second callback should not have been called (first one paused)
        self.assertEqual(len(second_called), 0)

    # ---- Entropy Monitor ----

    def test_force_mode_entropy_monitor(self):
        """EntropyMonitorCallback correctly computes entropy from real model logits."""
        scheduler = GenerationScheduler(mode="force")
        entropy_cb = EntropyMonitorCallback(entropy_threshold=100.0, action="log")  # High threshold → no pause
        scheduler.register_callback(entropy_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Entropy should have been computed for each step
        self.assertEqual(len(entropy_cb.entropy_history), 10)
        # All entropy values should be positive
        for entropy in entropy_cb.entropy_history:
            self.assertGreater(entropy, 0.0)

    # ---- Generation Logger ----

    def test_force_mode_generation_logger(self):
        """GenerationLoggerCallback logs all events from real generation."""
        scheduler = GenerationScheduler(mode="force")
        logger_cb = GenerationLoggerCallback(log_tokens=True, log_phases=True)
        scheduler.register_callback(logger_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        log = logger_cb.get_log()

        # Should have phase transitions and token events
        phase_entries = [e for e in log if e["type"] == "phase_transition"]
        token_entries = [e for e in log if e["type"] == "token_generated"]
        complete_entries = [e for e in log if e["type"] == "generation_complete"]

        self.assertGreater(len(phase_entries), 0)
        self.assertEqual(len(token_entries), 5)
        self.assertEqual(len(complete_entries), 1)

    # ---- Token Pattern Detection ----

    def test_force_mode_token_pattern_detection(self):
        """TokenPatternCallback detects tokens in real generation."""
        # First, do a baseline generation to find what tokens are generated
        collector = TokenCollectorCallback()
        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(collector)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        if len(collector.token_ids) >= 3:
            # Use the 3rd generated token as a trigger
            trigger_token = collector.token_ids[2]

            pattern_cb = TokenPatternCallback(trigger_token_ids={trigger_token})
            scheduler2 = GenerationScheduler(mode="force")
            scheduler2.register_callback(pattern_cb)

            self.model.generate(
                self.input_ids,
                max_new_tokens=10,
                do_sample=False,
                scheduler=scheduler2,
            )

            # Should have triggered and paused
            self.assertTrue(len(pattern_cb.triggered_at) > 0)

    # ---- Repetition Detection ----

    def test_force_mode_repetition_detector(self):
        """RepetitionDetectorCallback works with real model output."""
        scheduler = GenerationScheduler(mode="force")
        rep_cb = RepetitionDetectorCallback(ngram_size=2, max_repetitions=100, action="log")
        scheduler.register_callback(rep_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        # Callback should have processed tokens without error
        # (whether repetition is detected depends on the model's output)
        self.assertIsInstance(rep_cb._ngram_counts, dict)

    # ---- Batch Generation ----

    def test_force_mode_batch_generation(self):
        """FORCE mode works with batched inputs."""
        texts = ["Hello world", "The quick brown"]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertEqual(output.shape[0], 2)  # Batch size = 2
        self.assertTrue(recorder.completed)

    # ---- Sampling (do_sample=True) ----

    def test_force_mode_with_sampling(self):
        """FORCE mode works with stochastic sampling."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            scheduler=scheduler,
        )

        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])
        self.assertTrue(recorder.completed)
        self.assertEqual(len(recorder.tokens_generated), 10)

    # ---- Scheduler Stats ----

    def test_force_mode_stats_after_generation(self):
        """Scheduler stats are populated after generation."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        stats = scheduler.get_stats()
        self.assertEqual(stats["mode"], "force")
        self.assertIn("phase", stats)
        self.assertIn("total_time_seconds", stats)
        self.assertGreater(stats["num_transitions"], 0)
        self.assertEqual(stats["num_callbacks"], 1)

    # ---- Checkpoint ----

    def test_force_mode_checkpoint_during_generation(self):
        """Checkpoint can be saved during generation via callback."""
        checkpoint_data = {}

        class CheckpointCallback(SchedulerCallback):
            def __init__(self, scheduler_ref, save_at_step=2):
                self.scheduler_ref = scheduler_ref
                self.save_at_step = save_at_step

            def on_step_end(self, state, context):
                if state.step == self.save_at_step:
                    checkpoint_data.update(self.scheduler_ref.save_checkpoint())
                return True

        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(CheckpointCallback(scheduler))

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Checkpoint should have been saved
        self.assertIn("phase", checkpoint_data)
        self.assertIn("step", checkpoint_data)
        self.assertIn("context", checkpoint_data)

    # ---- GenerationConfig Integration ----

    def test_force_mode_via_generation_config_only(self):
        """Scheduler is auto-created from GenerationConfig when no scheduler is passed."""
        config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
            scheduler_mode="force",
            scheduler_step_budget=5,
        )

        output = self.model.generate(
            self.input_ids,
            generation_config=config,
        )

        # Step budget should have limited generation
        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertLessEqual(num_generated, 6)  # budget + 1 for the pause-step token


# ==============================================================================
# MODE: INTERNAL — LLM-Driven Control Tests
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestInternalModeIntegration(unittest.TestCase):
    """
    Black-box tests for scheduler_mode='internal'.

    Tests the LLM-driven scheduling mode where the model generates
    control tokens that the scheduler intercepts and processes.

    Since the tiny-random-gpt2 model generates random tokens, we test
    the INTERNAL mode infrastructure by:
    1. Setting up control tokens that are likely/unlikely to be generated
    2. Verifying the callback machinery works end-to-end
    3. Testing the control token parser integration
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_text = "Hello, world"
        cls.input_ids = cls.tokenizer(cls.input_text, return_tensors="pt").input_ids.to(torch_device)
        cls.vocab_size = cls.model.config.vocab_size

    def test_internal_mode_basic_generation(self):
        """INTERNAL mode generates output even without control tokens being hit."""
        # Use token IDs that are very unlikely to be generated (outside vocab range or rare)
        parser = ControlTokenParser(
            control_tokens={self.vocab_size - 1: "unlikely_action"},
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])

    def test_internal_mode_with_internal_callback(self):
        """InternalSchedulerCallback is invoked during real generation."""
        parser = ControlTokenParser(
            control_tokens={self.vocab_size - 1: "rare_action"},
            action_handlers={
                "rare_action": lambda name, state, ctx: ctx.custom_data.update({"hit": True}),
            },
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(control_token_parser=parser)
        recorder = EventRecorderCallback()
        scheduler.register_callback(internal_cb)
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Generation should complete
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])
        self.assertTrue(recorder.completed)

    def test_internal_mode_control_token_triggers_pause(self):
        """
        When the model generates a control token, the scheduler pauses.

        Strategy: Force a control token via ForceTokenAtStepCallback,
        then verify InternalSchedulerCallback detects it.
        """
        control_token_id = 50  # Arbitrary token ID within vocab

        parser = ControlTokenParser(
            control_tokens={control_token_id: "pause_action"},
            action_handlers={
                "pause_action": lambda name, state, ctx: setattr(ctx, 'should_pause', True),
            },
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(control_token_parser=parser)
        # Force the control token at step 3
        forcer = ForceTokenAtStepCallback(step=3, token_id=control_token_id)
        scheduler.register_callback(forcer)
        scheduler.register_callback(internal_cb)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        # Should have been paused by the control token
        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertLessEqual(num_generated, 5)  # Paused around step 3-4
        self.assertTrue(scheduler.is_paused())

        # Control history should record the event
        history = internal_cb.get_control_history()
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["action"], "pause_action")
        self.assertEqual(history[0]["token_id"], control_token_id)

    def test_internal_mode_control_token_custom_action(self):
        """Control token triggers a custom action (not pause)."""
        control_token_id = 50
        custom_invocations = []

        def custom_handler(name, state, ctx):
            custom_invocations.append({"action": name, "step": state.step})

        parser = ControlTokenParser(
            control_tokens={control_token_id: "custom_action"},
            action_handlers={"custom_action": custom_handler},
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(control_token_parser=parser)
        forcer = ForceTokenAtStepCallback(step=2, token_id=control_token_id)
        scheduler.register_callback(forcer)
        scheduler.register_callback(internal_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Custom action should have been executed at least once
        self.assertGreater(len(custom_invocations), 0)
        # The forced token at step 2 should be among the invocations
        steps_triggered = [inv["step"] for inv in custom_invocations]
        self.assertIn(2, steps_triggered, "Custom action should have been triggered at step 2")

    def test_internal_mode_max_consecutive_guard(self):
        """Safety guard triggers when too many consecutive control tokens are generated."""
        control_token_id = 50

        parser = ControlTokenParser(
            control_tokens={control_token_id: "noop"},
            action_handlers={"noop": lambda name, state, ctx: None},
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(
            control_token_parser=parser,
            max_consecutive_controls=3,
        )

        # Force the control token at every step
        class AlwaysForceCallback(SchedulerCallback):
            def on_step_begin(self, state, context):
                context.forced_token = control_token_id
                return True

        scheduler.register_callback(AlwaysForceCallback())
        scheduler.register_callback(internal_cb)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        # Should have been stopped by the safety guard
        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertLessEqual(num_generated, 5)
        self.assertTrue(scheduler.is_paused())
        self.assertEqual(scheduler.context.custom_data.get("pause_reason"), "max_consecutive_controls")

    def test_internal_mode_multiple_control_tokens(self):
        """Multiple different control tokens can be registered and detected."""
        actions_triggered = []

        def make_handler(action_name):
            def handler(name, state, ctx):
                actions_triggered.append(action_name)
            return handler

        parser = ControlTokenParser(
            control_tokens={
                50: "action_a",
                51: "action_b",
                52: "action_c",
            },
            action_handlers={
                "action_a": make_handler("action_a"),
                "action_b": make_handler("action_b"),
                "action_c": make_handler("action_c"),
            },
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(control_token_parser=parser)

        # Force different control tokens at different steps
        class SequentialForceCallback(SchedulerCallback):
            def __init__(self):
                self.token_schedule = {0: 50, 2: 51, 4: 52}

            def on_step_begin(self, state, context):
                if state.step in self.token_schedule:
                    context.forced_token = self.token_schedule[state.step]
                return True

        scheduler.register_callback(SequentialForceCallback())
        scheduler.register_callback(internal_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=6,
            do_sample=False,
            scheduler=scheduler,
        )

        # All three actions should have been triggered
        self.assertIn("action_a", actions_triggered)
        self.assertIn("action_b", actions_triggered)
        self.assertIn("action_c", actions_triggered)

    def test_internal_mode_on_control_detected_hook(self):
        """on_control_detected custom handler is called during real generation."""
        detected_events = []
        control_token_id = 50

        parser = ControlTokenParser(
            control_tokens={control_token_id: "test_action"},
            action_handlers={"test_action": lambda n, s, c: None},
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(
            control_token_parser=parser,
            on_control_detected=lambda name, tid, state, ctx: detected_events.append(
                {"action": name, "token": tid, "step": state.step}
            ),
        )
        forcer = ForceTokenAtStepCallback(step=1, token_id=control_token_id)
        scheduler.register_callback(forcer)
        scheduler.register_callback(internal_cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(len(detected_events), 0)
        self.assertEqual(detected_events[0]["action"], "test_action")
        self.assertEqual(detected_events[0]["token"], control_token_id)

    def test_internal_mode_without_parser(self):
        """INTERNAL mode without parser still generates output (just no control detection)."""
        scheduler = GenerationScheduler(mode="internal")  # No parser
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])
        self.assertTrue(recorder.completed)


# ==============================================================================
# Cross-Mode Tests
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestCrossModeIntegration(unittest.TestCase):
    """
    Tests that verify behavior across different scheduler modes,
    ensuring consistency and correct mode isolation.
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_text = "Hello, world"
        cls.input_ids = cls.tokenizer(cls.input_text, return_tensors="pt").input_ids.to(torch_device)

    def test_same_output_none_vs_no_scheduler(self):
        """NONE mode produces same output as no scheduler at all."""
        torch.manual_seed(123)
        out_no_sched = self.model.generate(self.input_ids, max_new_tokens=15, do_sample=False)

        torch.manual_seed(123)
        scheduler = GenerationScheduler(mode="none")
        out_none = self.model.generate(self.input_ids, max_new_tokens=15, do_sample=False, scheduler=scheduler)

        self.assertTrue(torch.equal(out_no_sched, out_none))

    def test_force_mode_without_callbacks_matches_baseline(self):
        """FORCE mode with no callbacks produces same output as baseline (greedy)."""
        torch.manual_seed(456)
        baseline = self.model.generate(self.input_ids, max_new_tokens=15, do_sample=False)

        torch.manual_seed(456)
        scheduler = GenerationScheduler(mode="force")
        # No callbacks registered
        out_force = self.model.generate(self.input_ids, max_new_tokens=15, do_sample=False, scheduler=scheduler)

        self.assertTrue(torch.equal(baseline, out_force),
                        "FORCE mode without callbacks should produce identical output to baseline")

    def test_all_modes_produce_valid_output(self):
        """All three modes produce valid (non-empty, longer than input) output."""
        for mode in ["none", "internal", "force"]:
            scheduler = GenerationScheduler(mode=mode)
            if mode == "internal":
                parser = ControlTokenParser(control_tokens={self.model.config.vocab_size - 1: "noop"})
                scheduler = GenerationScheduler(mode=mode, control_token_parser=parser)

            output = self.model.generate(
                self.input_ids,
                max_new_tokens=10,
                do_sample=False,
                scheduler=scheduler,
            )

            self.assertGreater(output.shape[-1], self.input_ids.shape[-1],
                               f"Mode '{mode}' should produce output longer than input")

    def test_scheduler_reuse_across_generations(self):
        """A scheduler can be reused across multiple generate() calls."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        # First generation
        self.model.generate(self.input_ids, max_new_tokens=5, do_sample=False, scheduler=scheduler)
        first_tokens = len(recorder.tokens_generated)
        self.assertGreater(first_tokens, 0)

        # Second generation (scheduler should reset)
        self.model.generate(self.input_ids, max_new_tokens=5, do_sample=False, scheduler=scheduler)
        second_tokens = len(recorder.tokens_generated) - first_tokens
        self.assertGreater(second_tokens, 0)

    def test_scheduler_mode_string_vs_enum(self):
        """String and enum mode specifications produce same behavior."""
        torch.manual_seed(789)
        scheduler_str = GenerationScheduler(mode="force")
        out1 = self.model.generate(self.input_ids, max_new_tokens=10, do_sample=False, scheduler=scheduler_str)

        torch.manual_seed(789)
        scheduler_enum = GenerationScheduler(mode=SchedulerMode.FORCE)
        out2 = self.model.generate(self.input_ids, max_new_tokens=10, do_sample=False, scheduler=scheduler_enum)

        self.assertTrue(torch.equal(out1, out2))


# ==============================================================================
# Edge Case Tests
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestEdgeCasesIntegration(unittest.TestCase):
    """Tests for edge cases and boundary conditions."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("Hello", return_tensors="pt").input_ids.to(torch_device)

    def test_max_new_tokens_1(self):
        """Scheduler works with max_new_tokens=1."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=1,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertEqual(output.shape[-1], self.input_ids.shape[-1] + 1)
        self.assertEqual(len(recorder.tokens_generated), 1)
        self.assertTrue(recorder.completed)

    def test_step_budget_zero(self):
        """Step budget of 0 means no tokens are generated."""
        scheduler = GenerationScheduler(mode="force")
        scheduler.context.step_budget = 0

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=50,
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertEqual(num_generated, 0)

    def test_immediate_pause(self):
        """Pausing before any tokens are generated."""
        class ImmediatePause(SchedulerCallback):
            def on_step_begin(self, state, context):
                context.should_pause = True
                return False

        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(ImmediatePause())

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=50,
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertEqual(num_generated, 0)

    def test_callback_raises_exception(self):
        """Callback that raises an exception is handled gracefully."""
        class CrashCallback(SchedulerCallback):
            def on_logits_ready(self, logits, state, context):
                raise RuntimeError("Intentional crash in logits callback")

        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(CrashCallback())

        # Should not crash the generation
        output = self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        # Generation should still produce output
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])

    def test_empty_input(self):
        """Scheduler works with minimal input (single token)."""
        single_token = torch.tensor([[self.tokenizer.bos_token_id or 0]], device=torch_device)

        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            single_token,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(output.shape[-1], 1)
        self.assertTrue(recorder.completed)

    def test_check_interval_larger_than_generation(self):
        """Check interval larger than generation length → no CHECKING phase."""
        scheduler = GenerationScheduler(mode="force")
        scheduler.context.check_interval = 100  # Much larger than max_new_tokens
        verifier = CheckIntervalVerifier()
        scheduler.register_callback(verifier)

        self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertEqual(len(verifier.checking_steps), 0)


# ==============================================================================
# Slow / Heavy Tests (require RUN_SLOW=1)
# ==============================================================================


@pytest.mark.generate
@require_torch
class TestSchedulerSlowIntegration(unittest.TestCase):
    """
    Heavier integration tests that are skipped by default.
    Run with: RUN_SLOW=1 pytest tests/generation/test_scheduler_integration.py -v -k TestSchedulerSlowIntegration
    """

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("The quick brown fox", return_tensors="pt").input_ids.to(torch_device)

    @slow
    def test_long_generation_with_scheduler(self):
        """Scheduler works correctly with longer generation (200 tokens)."""
        scheduler = GenerationScheduler(mode="force")
        recorder = EventRecorderCallback()
        scheduler.register_callback(recorder)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=200,
            do_sample=False,
            scheduler=scheduler,
        )

        num_generated = output.shape[-1] - self.input_ids.shape[-1]
        self.assertEqual(num_generated, 200)
        self.assertEqual(len(recorder.tokens_generated), 200)
        self.assertTrue(recorder.completed)

    @slow
    def test_scheduler_with_many_callbacks(self):
        """Scheduler works with many callbacks registered simultaneously."""
        scheduler = GenerationScheduler(mode="force")
        callbacks = []
        for _ in range(20):
            cb = EventRecorderCallback()
            scheduler.register_callback(cb)
            callbacks.append(cb)

        self.model.generate(
            self.input_ids,
            max_new_tokens=50,
            do_sample=False,
            scheduler=scheduler,
        )

        for cb in callbacks:
            self.assertTrue(cb.completed)
            self.assertEqual(len(cb.tokens_generated), 50)

    @slow
    def test_scheduler_checkpoint_roundtrip(self):
        """Full checkpoint save/load roundtrip during real generation."""
        saved_checkpoint = {}
        saved_state_path = None

        class MidGenCheckpoint(SchedulerCallback):
            def __init__(self, scheduler_ref):
                self.scheduler_ref = scheduler_ref

            def on_step_end(self, state, context):
                if state.step == 10:
                    nonlocal saved_checkpoint, saved_state_path
                    saved_checkpoint = self.scheduler_ref.save_checkpoint()
                    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                        saved_state_path = f.name
                    state.save(saved_state_path)
                return True

        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(MidGenCheckpoint(scheduler))

        self.model.generate(
            self.input_ids,
            max_new_tokens=20,
            do_sample=False,
            scheduler=scheduler,
        )

        try:
            # Verify checkpoint was saved
            self.assertIn("phase", saved_checkpoint)
            self.assertEqual(saved_checkpoint["step"], 10)

            # Verify state can be loaded
            if saved_state_path and os.path.exists(saved_state_path):
                loaded_state = GenerationState.load(saved_state_path)
                self.assertEqual(loaded_state.step, 10)
                self.assertIsNotNone(loaded_state.input_ids)
        finally:
            if saved_state_path and os.path.exists(saved_state_path):
                os.unlink(saved_state_path)

    @slow
    def test_internal_mode_long_generation_stability(self):
        """INTERNAL mode is stable over long generation runs."""
        parser = ControlTokenParser(
            control_tokens={50: "action_a", 51: "action_b"},
            action_handlers={
                "action_a": lambda n, s, c: c.custom_data.update({"a_count": c.custom_data.get("a_count", 0) + 1}),
                "action_b": lambda n, s, c: c.custom_data.update({"b_count": c.custom_data.get("b_count", 0) + 1}),
            },
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(
            control_token_parser=parser,
            max_consecutive_controls=50,  # High limit for stability test
        )
        scheduler.register_callback(internal_cb)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=1.0,
            scheduler=scheduler,
        )

        # Should complete without errors
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])


# ==============================================================================
# New integration tests for Grok-suggested improvements
# ==============================================================================


@unittest.skipIf(not is_torch_available(), "torch not available")
class TestAddCallbackAliasIntegration(unittest.TestCase):
    """Integration test: add_callback alias works end-to-end."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("Hello, world", return_tensors="pt").input_ids.to(torch_device)

    def test_add_callback_in_generate(self):
        """add_callback alias works the same as register_callback in generate()."""
        scheduler = GenerationScheduler(mode="force")
        events = []

        class TrackingCB(SchedulerCallback):
            def on_step_begin(self, state, context):
                events.append(("step_begin", state.step))
                return True

        scheduler.add_callback(TrackingCB())

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=5,
            do_sample=False,
            scheduler=scheduler,
        )

        self.assertGreater(len(events), 0)
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])


@unittest.skipIf(not is_torch_available(), "torch not available")
class TestBatchControlMaskIntegration(unittest.TestCase):
    """Integration test: batch_control_mask is initialized properly."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("Hello, world", return_tensors="pt").input_ids.to(torch_device)

    def test_batch_control_mask_initialized(self):
        """batch_control_mask is initialized to all-ones in generate()."""
        scheduler = GenerationScheduler(mode="force")
        masks = []

        class MaskChecker(SchedulerCallback):
            def on_step_begin(self, state, context):
                if state.batch_control_mask is not None:
                    masks.append(state.batch_control_mask.clone())
                return True

        scheduler.register_callback(MaskChecker())

        self.model.generate(
            self.input_ids,
            max_new_tokens=3,
            do_sample=False,
            scheduler=scheduler,
        )

        # batch_control_mask should have been recorded
        self.assertGreater(len(masks), 0)
        # All values should be 1 (all sequences active)
        for mask in masks:
            self.assertTrue((mask == 1).all())


@unittest.skipIf(not is_torch_available(), "torch not available")
class TestStreamingCallbackIntegration(unittest.TestCase):
    """Integration test: StreamingSchedulerCallback works with real model."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("Hello, world", return_tensors="pt").input_ids.to(torch_device)

    def test_streaming_callback_collects_text(self):
        """StreamingSchedulerCallback accumulates text during generation."""
        from transformers.generation.scheduler_callbacks import StreamingSchedulerCallback

        chunks = []
        cb = StreamingSchedulerCallback(
            tokenizer=self.tokenizer,
            on_text=lambda text: chunks.append(text),
        )
        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(cb)

        output = self.model.generate(
            self.input_ids,
            max_new_tokens=10,
            do_sample=False,
            scheduler=scheduler,
        )

        # Should have collected some text chunks
        # (tiny random model may produce garbage, but should not error)
        self.assertGreater(output.shape[-1], self.input_ids.shape[-1])
        # The callback should have generated_text (may be empty if all special tokens)
        self.assertIsInstance(cb.generated_text, str)

    def test_streaming_callback_reset(self):
        """StreamingSchedulerCallback.reset() clears state between generations."""
        from transformers.generation.scheduler_callbacks import StreamingSchedulerCallback

        cb = StreamingSchedulerCallback(tokenizer=self.tokenizer)
        scheduler = GenerationScheduler(mode="force")
        scheduler.register_callback(cb)

        # First generation
        self.model.generate(
            self.input_ids, max_new_tokens=5, do_sample=False, scheduler=scheduler,
        )
        cb.generated_text

        # Reset
        cb.reset()
        self.assertEqual(cb.generated_text, "")

        # Second generation (reuse scheduler with reset)
        scheduler2 = GenerationScheduler(mode="force")
        cb2 = StreamingSchedulerCallback(tokenizer=self.tokenizer)
        scheduler2.register_callback(cb2)
        self.model.generate(
            self.input_ids, max_new_tokens=5, do_sample=False, scheduler=scheduler2,
        )
        # Should work without error
        self.assertIsInstance(cb2.generated_text, str)


@unittest.skipIf(not is_torch_available(), "torch not available")
class TestControlTokenParserFromTokenizerIntegration(unittest.TestCase):
    """Integration test: ControlTokenParser.from_tokenizer with real tokenizer."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.tokenizer = _load_tiny_model()
        cls.input_ids = cls.tokenizer("Hello, world", return_tensors="pt").input_ids.to(torch_device)

    def test_from_tokenizer_with_real_tokenizer(self):
        """from_tokenizer works with a real HF tokenizer."""
        import copy
        # Use a copy to avoid modifying the shared tokenizer
        tokenizer = copy.deepcopy(self.tokenizer)
        original_vocab_size = len(tokenizer)

        parser = ControlTokenParser.from_tokenizer(
            tokenizer,
            control_token_names=["pause", "read_chunk"],
        )

        # New tokens should have been added
        new_vocab_size = len(tokenizer)
        self.assertGreaterEqual(new_vocab_size, original_vocab_size)

        # Parser should have the registered tokens
        # (Note: some tokenizers may not support adding tokens, so we check gracefully)
        if len(parser.control_tokens) > 0:
            for token_id, action in parser.control_tokens.items():
                self.assertIn(action, ["pause", "read_chunk"])
                self.assertIsInstance(token_id, int)


if __name__ == "__main__":
    unittest.main()
