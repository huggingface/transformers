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
Comprehensive tests for the GenerationScheduler, SchedulerCallback,
ControlTokenParser, configurable modes, and InternalSchedulerCallback.
"""

import os
import tempfile
import unittest

import torch

from transformers.generation.generation_scheduler import (
    ControlTokenParser,
    GenerationScheduler,
    SchedulerCallback,
    SchedulerContext,
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
# Helper Callbacks
# ==============================================================================


class CustomTestCallback(SchedulerCallback):
    """A test callback that records all events."""

    def __init__(self):
        super().__init__()
        self.events = []

    def on_phase_transition(self, from_phase, to_phase, state, context):
        self.events.append(("phase", from_phase.name, to_phase.name))
        return True

    def on_step_begin(self, state, context):
        self.events.append(("step_begin", state.step))
        return True

    def on_logits_ready(self, logits, state, context):
        self.events.append(("logits", state.step))
        return logits

    def on_token_generated(self, token_id, state, context):
        self.events.append(("token", token_id.item() if token_id.dim() == 0 else token_id[0].item()))
        return True

    def on_step_end(self, state, context):
        self.events.append(("step_end", state.step))
        return True

    def on_generation_complete(self, state):
        self.events.append(("complete", state.step))

    def on_error(self, error, state):
        self.events.append(("error", str(error)))


class PausingCallback(SchedulerCallback):
    """A callback that pauses at a specific step."""

    def __init__(self, pause_at_step: int):
        super().__init__()
        self.pause_at_step = pause_at_step

    def on_token_generated(self, token_id, state, context):
        if state.step >= self.pause_at_step:
            context.should_pause = True
            return False
        return True


# ==============================================================================
# SchedulerContext Tests
# ==============================================================================


class TestSchedulerContext(unittest.TestCase):
    """Tests for the SchedulerContext dataclass."""

    def test_default_context(self):
        """Default context has sensible defaults."""
        ctx = SchedulerContext()
        self.assertEqual(ctx.mode, SchedulerMode.NONE)
        self.assertFalse(ctx.should_pause)
        self.assertIsNone(ctx.forced_token)
        self.assertIsNone(ctx.tokens_to_inject)
        self.assertIsNone(ctx.logits_modifier)
        self.assertIsInstance(ctx.custom_data, dict)
        self.assertIsNone(ctx.step_budget)
        self.assertEqual(ctx.check_interval, 0)

    def test_context_with_mode(self):
        """Context can be initialized with a specific mode."""
        ctx = SchedulerContext(mode=SchedulerMode.FORCE)
        self.assertEqual(ctx.mode, SchedulerMode.FORCE)

        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)
        self.assertEqual(ctx.mode, SchedulerMode.INTERNAL)

    def test_context_mutation(self):
        """Context fields can be mutated."""
        ctx = SchedulerContext()
        ctx.should_pause = True
        ctx.forced_token = 42
        ctx.tokens_to_inject = [1, 2, 3]
        ctx.custom_data["key"] = "value"

        self.assertTrue(ctx.should_pause)
        self.assertEqual(ctx.forced_token, 42)
        self.assertEqual(ctx.tokens_to_inject, [1, 2, 3])
        self.assertEqual(ctx.custom_data["key"], "value")


# ==============================================================================
# ControlTokenParser Tests
# ==============================================================================


class TestControlTokenParser(unittest.TestCase):
    """Tests for the ControlTokenParser."""

    def setUp(self):
        self.parser = ControlTokenParser(
            control_tokens={32000: "pause", 32001: "read_chunk", 32002: "recall"},
            action_handlers={
                "pause": lambda name, state, ctx: setattr(ctx, "should_pause", True),
                "read_chunk": lambda name, state, ctx: ctx.custom_data.update({"action": "read_chunk"}),
            },
        )

    def test_is_control_token(self):
        """Correctly identifies control tokens."""
        self.assertTrue(self.parser.is_control_token(32000))
        self.assertTrue(self.parser.is_control_token(32001))
        self.assertFalse(self.parser.is_control_token(100))

    def test_get_action(self):
        """Maps token IDs to action names."""
        self.assertEqual(self.parser.get_action(32000), "pause")
        self.assertEqual(self.parser.get_action(32001), "read_chunk")
        self.assertEqual(self.parser.get_action(32002), "recall")
        self.assertIsNone(self.parser.get_action(100))

    def test_get_token_id(self):
        """Maps action names to token IDs."""
        self.assertEqual(self.parser.get_token_id("pause"), 32000)
        self.assertEqual(self.parser.get_token_id("read_chunk"), 32001)
        self.assertIsNone(self.parser.get_token_id("unknown"))

    def test_execute_with_handler(self):
        """Executes action handler correctly."""
        state = GenerationState()
        ctx = SchedulerContext()

        result = self.parser.execute(32000, state, ctx)
        self.assertTrue(result)
        self.assertTrue(ctx.should_pause)

    def test_execute_read_chunk(self):
        """Executes read_chunk handler correctly."""
        state = GenerationState()
        ctx = SchedulerContext()

        result = self.parser.execute(32001, state, ctx)
        self.assertTrue(result)
        self.assertEqual(ctx.custom_data["action"], "read_chunk")

    def test_execute_without_handler(self):
        """Returns False for action without handler (recall has no handler)."""
        state = GenerationState()
        ctx = SchedulerContext()

        result = self.parser.execute(32002, state, ctx)
        self.assertFalse(result)

    def test_execute_non_control_token(self):
        """Returns False for non-control token."""
        state = GenerationState()
        ctx = SchedulerContext()

        result = self.parser.execute(100, state, ctx)
        self.assertFalse(result)

    def test_register_action(self):
        """New actions can be registered dynamically."""
        handler_called = []
        self.parser.register_action(
            "new_action", 32003,
            handler=lambda name, state, ctx: handler_called.append(name)
        )

        self.assertTrue(self.parser.is_control_token(32003))
        self.assertEqual(self.parser.get_action(32003), "new_action")
        self.assertEqual(self.parser.get_token_id("new_action"), 32003)

        state = GenerationState()
        ctx = SchedulerContext()
        self.parser.execute(32003, state, ctx)
        self.assertEqual(handler_called, ["new_action"])

    def test_strip_control_tokens_flag(self):
        """strip_control_tokens flag works correctly."""
        parser = ControlTokenParser(
            control_tokens={32000: "pause"},
            strip_control_tokens=True,
        )
        self.assertTrue(parser.strip_control_tokens)

        parser_no_strip = ControlTokenParser(
            control_tokens={32000: "pause"},
            strip_control_tokens=False,
        )
        self.assertFalse(parser_no_strip.strip_control_tokens)

    def test_empty_parser(self):
        """Empty parser works correctly."""
        parser = ControlTokenParser()
        self.assertFalse(parser.is_control_token(100))
        self.assertIsNone(parser.get_action(100))


# ==============================================================================
# SchedulerCallback Tests
# ==============================================================================


class TestSchedulerCallback(unittest.TestCase):
    """Tests for the base SchedulerCallback."""

    def test_default_callbacks_return_true(self):
        """Base callback methods return True (continue) by default."""
        cb = SchedulerCallback()
        state = GenerationState()
        ctx = SchedulerContext()

        self.assertTrue(cb.on_phase_transition(
            GenerationPhase.IDLE, GenerationPhase.INIT, state, ctx
        ))
        self.assertTrue(cb.on_step_begin(state, ctx))
        self.assertTrue(cb.on_token_generated(torch.tensor([1]), state, ctx))
        self.assertTrue(cb.on_step_end(state, ctx))

    def test_logits_passthrough(self):
        """Base on_logits_ready returns logits unchanged."""
        cb = SchedulerCallback()
        logits = torch.randn(2, 100)
        state = GenerationState()
        ctx = SchedulerContext()

        result = cb.on_logits_ready(logits, state, ctx)
        self.assertTrue(torch.equal(result, logits))

    def test_complete_and_error_dont_crash(self):
        """on_generation_complete and on_error don't raise."""
        cb = SchedulerCallback()
        state = GenerationState()
        cb.on_generation_complete(state)
        cb.on_error(RuntimeError("test"), state)


# ==============================================================================
# GenerationScheduler Mode Tests
# ==============================================================================


class TestGenerationSchedulerModes(unittest.TestCase):
    """Tests for the three scheduler modes: NONE, INTERNAL, FORCE."""

    def test_default_mode_is_none(self):
        """Default scheduler mode is NONE."""
        scheduler = GenerationScheduler()
        self.assertEqual(scheduler.mode, SchedulerMode.NONE)
        self.assertFalse(scheduler.is_enabled)

    def test_mode_from_string(self):
        """Scheduler mode can be set from string."""
        scheduler = GenerationScheduler(mode="none")
        self.assertEqual(scheduler.mode, SchedulerMode.NONE)

        scheduler = GenerationScheduler(mode="internal")
        self.assertEqual(scheduler.mode, SchedulerMode.INTERNAL)

        scheduler = GenerationScheduler(mode="force")
        self.assertEqual(scheduler.mode, SchedulerMode.FORCE)

    def test_mode_from_enum(self):
        """Scheduler mode can be set from enum."""
        scheduler = GenerationScheduler(mode=SchedulerMode.FORCE)
        self.assertEqual(scheduler.mode, SchedulerMode.FORCE)
        self.assertTrue(scheduler.is_enabled)

    def test_none_mode_is_noop(self):
        """NONE mode scheduler does not activate."""
        scheduler = GenerationScheduler(mode="none")
        scheduler._activate()
        self.assertFalse(scheduler.is_active)

    def test_force_mode_activates(self):
        """FORCE mode scheduler activates correctly."""
        scheduler = GenerationScheduler(mode="force")
        scheduler._activate()
        self.assertTrue(scheduler.is_active)
        self.assertEqual(scheduler.context.mode, SchedulerMode.FORCE)

    def test_internal_mode_activates(self):
        """INTERNAL mode scheduler activates correctly."""
        parser = ControlTokenParser(control_tokens={32000: "pause"})
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        scheduler._activate()
        self.assertTrue(scheduler.is_active)
        self.assertEqual(scheduler.context.mode, SchedulerMode.INTERNAL)

    def test_internal_mode_without_parser_warns(self):
        """INTERNAL mode without parser logs a warning (but doesn't crash)."""
        # Should not raise, just warn
        scheduler = GenerationScheduler(mode="internal")
        self.assertEqual(scheduler.mode, SchedulerMode.INTERNAL)
        self.assertIsNone(scheduler.control_token_parser)

    def test_context_mode_matches_scheduler(self):
        """Context mode is synchronized with scheduler mode."""
        for mode_str in ["none", "internal", "force"]:
            scheduler = GenerationScheduler(mode=mode_str)
            self.assertEqual(scheduler.context.mode.value, mode_str)


# ==============================================================================
# GenerationScheduler Core Tests
# ==============================================================================


class TestGenerationScheduler(unittest.TestCase):
    """Tests for the GenerationScheduler core functionality."""

    def setUp(self):
        self.scheduler = GenerationScheduler(mode="force")

    def test_initial_state(self):
        """Scheduler starts in IDLE phase, inactive."""
        self.assertEqual(self.scheduler.phase, GenerationPhase.IDLE)
        self.assertFalse(self.scheduler.is_active)
        self.assertFalse(self.scheduler.is_complete())
        self.assertFalse(self.scheduler.is_paused())

    def test_pause_resume(self):
        """Pause and resume work correctly."""
        self.scheduler.pause()
        self.assertTrue(self.scheduler.is_paused())
        self.assertTrue(self.scheduler.context.should_pause)

        self.scheduler.resume()
        self.assertFalse(self.scheduler.is_paused())
        self.assertFalse(self.scheduler.context.should_pause)

    def test_force_token(self):
        """Force token sets and consumes correctly."""
        self.scheduler.force_next_token(42)
        self.assertTrue(self.scheduler._has_forced_token())
        self.assertEqual(self.scheduler._consume_forced_token(), 42)
        self.assertFalse(self.scheduler._has_forced_token())
        self.assertIsNone(self.scheduler._consume_forced_token())

    def test_inject_tokens(self):
        """Token injection sets and consumes correctly."""
        self.scheduler.inject_tokens([10, 20, 30])
        self.assertTrue(self.scheduler._has_injection())

        tokens = self.scheduler._consume_injection()
        self.assertEqual(tokens, [10, 20, 30])
        self.assertFalse(self.scheduler._has_injection())
        self.assertIsNone(self.scheduler._consume_injection())

    def test_register_callback(self):
        """Callbacks can be registered and removed."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)
        self.assertEqual(len(self.scheduler.callbacks), 1)

        self.scheduler.remove_callback(cb)
        self.assertEqual(len(self.scheduler.callbacks), 0)

    def test_register_invalid_callback(self):
        """Registering a non-SchedulerCallback raises TypeError."""
        with self.assertRaises(TypeError):
            self.scheduler.register_callback("not_a_callback")

    def test_clear_callbacks(self):
        """clear_callbacks removes all callbacks."""
        self.scheduler.register_callback(CustomTestCallback())
        self.scheduler.register_callback(CustomTestCallback())
        self.assertEqual(len(self.scheduler.callbacks), 2)

        self.scheduler.clear_callbacks()
        self.assertEqual(len(self.scheduler.callbacks), 0)

    def test_activate_deactivate(self):
        """Activation and deactivation lifecycle."""
        self.scheduler._activate()
        self.assertTrue(self.scheduler.is_active)
        self.assertEqual(self.scheduler.phase, GenerationPhase.IDLE)

        self.scheduler._deactivate()
        self.assertFalse(self.scheduler.is_active)

    def test_notify_phase_transition(self):
        """Phase transition notifications work."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)

        state = GenerationState()
        result = self.scheduler._notify_phase_transition(
            GenerationPhase.IDLE, GenerationPhase.INIT, state
        )

        self.assertTrue(result)
        self.assertEqual(self.scheduler.phase, GenerationPhase.INIT)
        self.assertEqual(len(cb.events), 1)
        self.assertEqual(cb.events[0], ("phase", "IDLE", "INIT"))

    def test_notify_phase_transition_callback_pause(self):
        """Phase transition can be paused by callback."""
        class PausingPhaseCallback(SchedulerCallback):
            def on_phase_transition(self, from_phase, to_phase, state, context):
                return False  # Always pause

        self.scheduler.register_callback(PausingPhaseCallback())
        state = GenerationState()
        result = self.scheduler._notify_phase_transition(
            GenerationPhase.IDLE, GenerationPhase.INIT, state
        )
        self.assertFalse(result)

    def test_notify_logits_ready(self):
        """Logits notification passes through callbacks."""
        class DoublingCallback(SchedulerCallback):
            def on_logits_ready(self, logits, state, context):
                return logits * 2

        self.scheduler.register_callback(DoublingCallback())
        logits = torch.ones(2, 10)
        state = GenerationState()

        result = self.scheduler._notify_logits_ready(logits, state)
        self.assertTrue(torch.allclose(result, torch.ones(2, 10) * 2))

    def test_notify_logits_ready_chaining(self):
        """Multiple logits callbacks are chained."""
        class AddOneCallback(SchedulerCallback):
            def on_logits_ready(self, logits, state, context):
                return logits + 1

        self.scheduler.register_callback(AddOneCallback())
        self.scheduler.register_callback(AddOneCallback())

        logits = torch.zeros(1, 5)
        state = GenerationState()
        result = self.scheduler._notify_logits_ready(logits, state)
        self.assertTrue(torch.allclose(result, torch.ones(1, 5) * 2))

    def test_notify_token_generated(self):
        """Token generation notification works."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)

        state = GenerationState()
        token = torch.tensor([42])
        result = self.scheduler._notify_token_generated(token, state)

        self.assertTrue(result)
        self.assertEqual(len(cb.events), 1)
        self.assertEqual(cb.events[0], ("token", 42))

    def test_notify_token_generated_pause(self):
        """Token generation can trigger pause."""
        self.scheduler.register_callback(PausingCallback(pause_at_step=3))

        state = GenerationState(step=3)
        token = torch.tensor([1])
        result = self.scheduler._notify_token_generated(token, state)
        self.assertFalse(result)

    def test_notify_step_begin_end(self):
        """Step begin/end notifications work."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)

        state = GenerationState(step=5)
        self.scheduler._notify_step_begin(state)
        self.scheduler._notify_step_end(state)

        self.assertEqual(len(cb.events), 2)
        self.assertEqual(cb.events[0], ("step_begin", 5))
        self.assertEqual(cb.events[1], ("step_end", 5))

    def test_notify_generation_complete(self):
        """Completion notification works."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)

        state = GenerationState(step=10)
        self.scheduler._notify_generation_complete(state)

        self.assertEqual(len(cb.events), 1)
        self.assertEqual(cb.events[0], ("complete", 10))

    def test_notify_error(self):
        """Error notification works."""
        cb = CustomTestCallback()
        self.scheduler.register_callback(cb)

        state = GenerationState()
        self.scheduler._notify_error(RuntimeError("test error"), state)

        self.assertEqual(len(cb.events), 1)
        self.assertEqual(cb.events[0], ("error", "test error"))

    def test_should_check(self):
        """Check interval logic works."""
        self.scheduler.context.check_interval = 5

        self.assertFalse(self.scheduler._should_check(0))
        self.assertFalse(self.scheduler._should_check(1))
        self.assertTrue(self.scheduler._should_check(5))
        self.assertTrue(self.scheduler._should_check(10))
        self.assertFalse(self.scheduler._should_check(7))

    def test_should_check_disabled(self):
        """Check interval of 0 disables checking."""
        self.scheduler.context.check_interval = 0
        self.assertFalse(self.scheduler._should_check(0))
        self.assertFalse(self.scheduler._should_check(5))
        self.assertFalse(self.scheduler._should_check(100))

    def test_check_step_budget(self):
        """Step budget enforcement works."""
        self.scheduler.context.step_budget = 10

        self.assertTrue(self.scheduler._check_step_budget(0))
        self.assertTrue(self.scheduler._check_step_budget(9))
        self.assertFalse(self.scheduler._check_step_budget(10))
        self.assertFalse(self.scheduler._check_step_budget(15))

    def test_check_step_budget_unlimited(self):
        """None budget means unlimited."""
        self.scheduler.context.step_budget = None
        self.assertTrue(self.scheduler._check_step_budget(0))
        self.assertTrue(self.scheduler._check_step_budget(1000000))

    def test_empty_scheduler_zero_overhead(self):
        """Scheduler with no callbacks has minimal overhead."""
        state = GenerationState()
        logits = torch.randn(1, 100)

        # All notifications should return True / pass through
        self.assertTrue(self.scheduler._notify_phase_transition(
            GenerationPhase.IDLE, GenerationPhase.INIT, state
        ))
        self.assertTrue(self.scheduler._notify_step_begin(state))
        result = self.scheduler._notify_logits_ready(logits, state)
        self.assertTrue(torch.equal(result, logits))
        self.assertTrue(self.scheduler._notify_token_generated(torch.tensor([1]), state))
        self.assertTrue(self.scheduler._notify_step_end(state))

    def test_callback_exception_handling(self):
        """Callbacks that raise exceptions are handled gracefully."""
        class CrashingCallback(SchedulerCallback):
            def on_step_begin(self, state, context):
                raise RuntimeError("Callback crashed!")

        self.scheduler.register_callback(CrashingCallback())
        state = GenerationState()

        # Should not raise, but return False
        result = self.scheduler._notify_step_begin(state)
        self.assertFalse(result)


# ==============================================================================
# Internal Mode Token Handling Tests
# ==============================================================================


class TestInternalModeTokenHandling(unittest.TestCase):
    """Tests for INTERNAL mode control token handling."""

    def setUp(self):
        self.parser = ControlTokenParser(
            control_tokens={32000: "pause", 32001: "read_chunk"},
            action_handlers={
                "pause": lambda name, state, ctx: setattr(ctx, "should_pause", True),
                "read_chunk": lambda name, state, ctx: ctx.custom_data.update({"chunk_action": True}),
            },
        )
        self.scheduler = GenerationScheduler(mode="internal", control_token_parser=self.parser)

    def test_handle_internal_token_control(self):
        """Control tokens are handled in INTERNAL mode."""
        state = GenerationState()
        result = self.scheduler._handle_internal_token(32000, state)
        self.assertTrue(result)
        self.assertTrue(self.scheduler.context.should_pause)

    def test_handle_internal_token_normal(self):
        """Normal tokens are not handled in INTERNAL mode."""
        state = GenerationState()
        result = self.scheduler._handle_internal_token(100, state)
        self.assertFalse(result)

    def test_handle_internal_token_wrong_mode(self):
        """Tokens are not handled in non-INTERNAL modes."""
        scheduler = GenerationScheduler(mode="force")
        state = GenerationState()
        result = scheduler._handle_internal_token(32000, state)
        self.assertFalse(result)

    def test_should_strip_token(self):
        """Control tokens should be stripped in INTERNAL mode."""
        self.assertTrue(self.scheduler._should_strip_token(32000))
        self.assertFalse(self.scheduler._should_strip_token(100))

    def test_should_not_strip_when_disabled(self):
        """Tokens not stripped when strip_control_tokens is False."""
        parser = ControlTokenParser(
            control_tokens={32000: "pause"},
            strip_control_tokens=False,
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        self.assertFalse(scheduler._should_strip_token(32000))


# ==============================================================================
# InternalSchedulerCallback Tests
# ==============================================================================


class TestInternalSchedulerCallback(unittest.TestCase):
    """Tests for the InternalSchedulerCallback."""

    def setUp(self):
        self.parser = ControlTokenParser(
            control_tokens={32000: "pause", 32001: "read_chunk", 32002: "summary_done"},
            action_handlers={
                "pause": lambda name, state, ctx: setattr(ctx, "should_pause", True),
                "read_chunk": lambda name, state, ctx: ctx.custom_data.update({"action": "read_chunk"}),
                "summary_done": lambda name, state, ctx: setattr(ctx, "should_pause", True),
            },
        )
        self.callback = InternalSchedulerCallback(control_token_parser=self.parser)

    def test_control_token_detection(self):
        """Detects control tokens and executes actions."""
        state = GenerationState(step=5)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        result = self.callback.on_token_generated(torch.tensor([32001]), state, ctx)
        self.assertTrue(result)  # read_chunk doesn't pause
        self.assertEqual(ctx.custom_data["action"], "read_chunk")
        self.assertEqual(len(self.callback.control_history), 1)

    def test_pause_on_control_token(self):
        """Pauses when control token handler sets should_pause."""
        state = GenerationState(step=3)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        result = self.callback.on_token_generated(torch.tensor([32000]), state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)

    def test_normal_token_passthrough(self):
        """Normal tokens pass through without action."""
        state = GenerationState(step=1)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        result = self.callback.on_token_generated(torch.tensor([100]), state, ctx)
        self.assertTrue(result)
        self.assertEqual(len(self.callback.control_history), 0)

    def test_non_internal_mode_passthrough(self):
        """Callback is inactive in non-INTERNAL mode."""
        state = GenerationState(step=1)
        ctx = SchedulerContext(mode=SchedulerMode.FORCE)

        result = self.callback.on_token_generated(torch.tensor([32000]), state, ctx)
        self.assertTrue(result)  # Passes through
        self.assertEqual(len(self.callback.control_history), 0)

    def test_max_consecutive_controls_guard(self):
        """Safety guard triggers after max consecutive control tokens."""
        callback = InternalSchedulerCallback(
            control_token_parser=ControlTokenParser(
                control_tokens={32000: "noop"},
                action_handlers={"noop": lambda n, s, c: None},
            ),
            max_consecutive_controls=3,
        )
        state = GenerationState(step=0)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        # First 2 should pass
        for _ in range(2):
            result = callback.on_token_generated(torch.tensor([32000]), state, ctx)
            self.assertTrue(result)

        # Third should trigger safety guard
        result = callback.on_token_generated(torch.tensor([32000]), state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)
        self.assertEqual(ctx.custom_data["pause_reason"], "max_consecutive_controls")

    def test_consecutive_counter_resets_on_normal_token(self):
        """Consecutive control counter resets when a normal token is generated."""
        callback = InternalSchedulerCallback(
            control_token_parser=ControlTokenParser(
                control_tokens={32000: "noop"},
                action_handlers={"noop": lambda n, s, c: None},
            ),
            max_consecutive_controls=3,
        )
        state = GenerationState(step=0)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        # Two control tokens
        callback.on_token_generated(torch.tensor([32000]), state, ctx)
        callback.on_token_generated(torch.tensor([32000]), state, ctx)

        # Normal token resets counter
        callback.on_token_generated(torch.tensor([100]), state, ctx)

        # Two more control tokens should be fine
        result = callback.on_token_generated(torch.tensor([32000]), state, ctx)
        self.assertTrue(result)
        result = callback.on_token_generated(torch.tensor([32000]), state, ctx)
        self.assertTrue(result)

    def test_custom_on_control_detected(self):
        """Custom on_control_detected handler is called."""
        detected = []
        callback = InternalSchedulerCallback(
            control_token_parser=self.parser,
            on_control_detected=lambda name, tid, state, ctx: detected.append(name),
        )
        state = GenerationState(step=0)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        callback.on_token_generated(torch.tensor([32001]), state, ctx)
        self.assertEqual(detected, ["read_chunk"])

    def test_control_history(self):
        """Control history is recorded correctly."""
        state = GenerationState(step=5)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        self.callback.on_token_generated(torch.tensor([32001]), state, ctx)

        history = self.callback.get_control_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["step"], 5)
        self.assertEqual(history[0]["token_id"], 32001)
        self.assertEqual(history[0]["action"], "read_chunk")

    def test_clear_history(self):
        """History can be cleared."""
        state = GenerationState(step=0)
        ctx = SchedulerContext(mode=SchedulerMode.INTERNAL)

        self.callback.on_token_generated(torch.tensor([32001]), state, ctx)
        self.assertEqual(len(self.callback.get_control_history()), 1)

        self.callback.clear_history()
        self.assertEqual(len(self.callback.get_control_history()), 0)


# ==============================================================================
# Checkpoint Tests
# ==============================================================================


class TestSchedulerCheckpoint(unittest.TestCase):
    """Tests for scheduler checkpoint save/load."""

    def test_save_load_checkpoint_dict(self):
        """Checkpoint save/load via dict works."""
        scheduler = GenerationScheduler(mode="force")
        scheduler._activate()
        scheduler._notify_phase_transition(
            GenerationPhase.IDLE, GenerationPhase.INIT, GenerationState()
        )
        scheduler.context.custom_data["test"] = "value"
        scheduler.context.step_budget = 50

        checkpoint = scheduler.save_checkpoint()

        # Create a new scheduler and load
        new_scheduler = GenerationScheduler(mode="force")
        new_scheduler.load_checkpoint(checkpoint)

        self.assertEqual(new_scheduler.phase, GenerationPhase.INIT)
        self.assertEqual(new_scheduler.context.custom_data["test"], "value")
        self.assertEqual(new_scheduler.context.step_budget, 50)

    def test_save_load_checkpoint_file(self):
        """Checkpoint save/load via file works."""
        scheduler = GenerationScheduler(mode="force")
        state = GenerationState(
            phase=GenerationPhase.DECODING,
            step=5,
            input_ids=torch.randint(0, 100, (1, 10)),
        )
        scheduler.state_machine.current_state = state

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            scheduler.save_checkpoint(path=path)

            new_scheduler = GenerationScheduler(mode="force")
            new_scheduler.load_checkpoint(path)

            self.assertEqual(new_scheduler.phase, GenerationPhase.DECODING)
            loaded_state = new_scheduler.get_state()
            self.assertEqual(loaded_state.step, 5)
            self.assertTrue(torch.equal(loaded_state.input_ids, state.input_ids.cpu()))
        finally:
            os.unlink(path)


# ==============================================================================
# Stats and Repr Tests
# ==============================================================================


class TestSchedulerStatsAndRepr(unittest.TestCase):
    """Tests for scheduler statistics and repr."""

    def test_get_stats_includes_mode(self):
        """Stats contain mode information."""
        scheduler = GenerationScheduler(mode="force")
        scheduler._activate()
        stats = scheduler.get_stats()

        self.assertIn("mode", stats)
        self.assertEqual(stats["mode"], "force")
        self.assertIn("phase", stats)
        self.assertIn("step", stats)
        self.assertIn("total_time_seconds", stats)
        self.assertIn("num_callbacks", stats)
        self.assertIn("num_transitions", stats)
        self.assertIn("is_paused", stats)

    def test_repr_includes_mode(self):
        """Repr contains mode info."""
        scheduler = GenerationScheduler(mode="force")
        repr_str = repr(scheduler)
        self.assertIn("GenerationScheduler", repr_str)
        self.assertIn("force", repr_str)
        self.assertIn("IDLE", repr_str)


# ==============================================================================
# Full Lifecycle Simulation Tests
# ==============================================================================


class TestFullLifecycleSimulation(unittest.TestCase):
    """Simulate complete generation lifecycles with different modes."""

    def test_force_mode_lifecycle(self):
        """Simulate a complete FORCE mode generation lifecycle."""
        scheduler = GenerationScheduler(mode="force")
        cb = CustomTestCallback()
        scheduler.register_callback(cb)
        scheduler._activate()

        state = GenerationState()

        # IDLE → INIT → PREFILL → DECODING
        scheduler._notify_phase_transition(GenerationPhase.IDLE, GenerationPhase.INIT, state)
        scheduler._notify_phase_transition(GenerationPhase.INIT, GenerationPhase.PREFILL, state)
        scheduler._notify_phase_transition(GenerationPhase.PREFILL, GenerationPhase.DECODING, state)

        # Simulate 3 decode steps
        for step in range(3):
            state.step = step
            scheduler._notify_step_begin(state)
            logits = torch.randn(1, 100)
            scheduler._notify_logits_ready(logits, state)
            token = torch.tensor([step + 10])
            scheduler._notify_token_generated(token, state)
            scheduler._notify_step_end(state)

        # DECODING → POSTPROCESS → COMPLETE
        scheduler._notify_phase_transition(GenerationPhase.DECODING, GenerationPhase.POSTPROCESS, state)
        scheduler._notify_phase_transition(GenerationPhase.POSTPROCESS, GenerationPhase.COMPLETE, state)
        scheduler._notify_generation_complete(state)
        scheduler._deactivate()

        # Verify events
        phase_events = [e for e in cb.events if e[0] == "phase"]
        self.assertEqual(len(phase_events), 5)

        token_events = [e for e in cb.events if e[0] == "token"]
        self.assertEqual(len(token_events), 3)
        self.assertEqual(token_events[0][1], 10)
        self.assertEqual(token_events[1][1], 11)
        self.assertEqual(token_events[2][1], 12)

    def test_internal_mode_lifecycle_with_control_tokens(self):
        """Simulate INTERNAL mode with control token detection."""
        parser = ControlTokenParser(
            control_tokens={32000: "pause"},
            action_handlers={
                "pause": lambda name, state, ctx: setattr(ctx, "should_pause", True),
            },
        )
        scheduler = GenerationScheduler(mode="internal", control_token_parser=parser)
        internal_cb = InternalSchedulerCallback(control_token_parser=parser)
        scheduler.register_callback(internal_cb)
        scheduler._activate()

        state = GenerationState()

        # Simulate generation with a control token
        scheduler._notify_phase_transition(GenerationPhase.IDLE, GenerationPhase.INIT, state)
        scheduler._notify_phase_transition(GenerationPhase.INIT, GenerationPhase.PREFILL, state)
        scheduler._notify_phase_transition(GenerationPhase.PREFILL, GenerationPhase.DECODING, state)

        # Step 0: normal token
        state.step = 0
        scheduler._notify_step_begin(state)
        result = scheduler._notify_token_generated(torch.tensor([100]), state)
        self.assertTrue(result)
        scheduler._notify_step_end(state)

        # Step 1: control token → should pause
        state.step = 1
        scheduler._notify_step_begin(state)
        result = scheduler._notify_token_generated(torch.tensor([32000]), state)
        self.assertFalse(result)  # Paused by control token

        self.assertTrue(scheduler.is_paused())
        self.assertEqual(len(internal_cb.control_history), 1)
        self.assertEqual(internal_cb.control_history[0]["action"], "pause")

    def test_none_mode_no_interference(self):
        """NONE mode scheduler does not interfere with generation."""
        scheduler = GenerationScheduler(mode="none")
        self.assertFalse(scheduler.is_enabled)
        self.assertFalse(scheduler.is_active)

        # Activate should be a no-op
        scheduler._activate()
        self.assertFalse(scheduler.is_active)


# ==============================================================================
# Preset Callback Tests
# ==============================================================================


class TestEntropyMonitorCallback(unittest.TestCase):
    """Tests for the EntropyMonitorCallback."""

    def test_entropy_computation(self):
        """Entropy is computed correctly."""
        cb = EntropyMonitorCallback()

        uniform_logits = torch.zeros(1, 100)
        state = GenerationState(step=1)
        ctx = SchedulerContext()
        cb.on_logits_ready(uniform_logits, state, ctx)
        self.assertGreater(cb.entropy_history[-1], 0)

        peaked_logits = torch.full((1, 100), -100.0)
        peaked_logits[0, 0] = 100.0
        cb.on_logits_ready(peaked_logits, state, ctx)
        self.assertLess(cb.entropy_history[-1], cb.entropy_history[0])

    def test_pause_on_high_entropy(self):
        """Pauses when entropy exceeds threshold."""
        cb = EntropyMonitorCallback(entropy_threshold=1.0, action="pause", min_step=0)
        logits = torch.zeros(1, 1000)
        state = GenerationState(step=1)
        ctx = SchedulerContext()

        cb.on_logits_ready(logits, state, ctx)
        result = cb.on_token_generated(torch.tensor([1]), state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)

    def test_no_pause_on_low_entropy(self):
        """Does not pause when entropy is below threshold."""
        cb = EntropyMonitorCallback(entropy_threshold=10.0, action="pause", min_step=0)
        logits = torch.full((1, 100), -100.0)
        logits[0, 0] = 100.0
        state = GenerationState(step=1)
        ctx = SchedulerContext()

        cb.on_logits_ready(logits, state, ctx)
        result = cb.on_token_generated(torch.tensor([1]), state, ctx)
        self.assertTrue(result)
        self.assertFalse(ctx.should_pause)


class TestTokenPatternCallback(unittest.TestCase):
    """Tests for the TokenPatternCallback."""

    def test_single_token_trigger(self):
        """Detects single trigger tokens."""
        cb = TokenPatternCallback(trigger_token_ids={42, 99})
        state = GenerationState()
        ctx = SchedulerContext()

        result = cb.on_token_generated(torch.tensor([10]), state, ctx)
        self.assertTrue(result)

        result = cb.on_token_generated(torch.tensor([42]), state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)

    def test_sequence_trigger(self):
        """Detects token sequence triggers."""
        cb = TokenPatternCallback(trigger_sequences=[[1, 2, 3]])
        state = GenerationState()
        ctx = SchedulerContext()

        cb.on_token_generated(torch.tensor([1]), state, ctx)
        self.assertFalse(ctx.should_pause)
        cb.on_token_generated(torch.tensor([2]), state, ctx)
        self.assertFalse(ctx.should_pause)
        result = cb.on_token_generated(torch.tensor([3]), state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)


class TestGenerationLoggerCallback(unittest.TestCase):
    """Tests for the GenerationLoggerCallback."""

    def test_logs_phase_transitions(self):
        """Phase transitions are logged."""
        cb = GenerationLoggerCallback()
        state = GenerationState(step=0)
        ctx = SchedulerContext()

        cb.on_phase_transition(GenerationPhase.IDLE, GenerationPhase.INIT, state, ctx)

        log = cb.get_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["type"], "phase_transition")

    def test_clear_log(self):
        """Log can be cleared."""
        cb = GenerationLoggerCallback()
        state = GenerationState()
        ctx = SchedulerContext()
        cb.on_token_generated(torch.tensor([1]), state, ctx)
        self.assertEqual(len(cb.get_log()), 1)
        cb.clear_log()
        self.assertEqual(len(cb.get_log()), 0)


class TestStepBudgetCallback(unittest.TestCase):
    """Tests for the StepBudgetCallback."""

    def test_budget_enforcement(self):
        """Pauses when budget is exceeded."""
        cb = StepBudgetCallback(max_steps=10)
        ctx = SchedulerContext()

        state = GenerationState(step=5)
        result = cb.on_step_end(state, ctx)
        self.assertTrue(result)

        state = GenerationState(step=10)
        result = cb.on_step_end(state, ctx)
        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)


class TestRepetitionDetectorCallback(unittest.TestCase):
    """Tests for the RepetitionDetectorCallback."""

    def test_detects_repetition(self):
        """Detects repeated n-grams."""
        cb = RepetitionDetectorCallback(ngram_size=2, max_repetitions=3, action="pause")
        state = GenerationState()
        ctx = SchedulerContext()

        for _ in range(3):
            cb.on_token_generated(torch.tensor([1]), state, ctx)
            result = cb.on_token_generated(torch.tensor([2]), state, ctx)

        self.assertFalse(result)
        self.assertTrue(ctx.should_pause)

    def test_no_false_positive(self):
        """Does not trigger on non-repetitive sequences."""
        cb = RepetitionDetectorCallback(ngram_size=2, max_repetitions=3)
        state = GenerationState()
        ctx = SchedulerContext()

        for i in range(10):
            result = cb.on_token_generated(torch.tensor([i]), state, ctx)
            self.assertTrue(result)
        self.assertFalse(ctx.should_pause)


# ==============================================================================
# Multiple Callback Composition Tests
# ==============================================================================


class TestMultipleCallbackComposition(unittest.TestCase):
    """Tests for composing multiple callbacks."""

    def test_multiple_callbacks_all_called(self):
        """All registered callbacks receive events."""
        scheduler = GenerationScheduler(mode="force")
        cb1 = CustomTestCallback()
        cb2 = CustomTestCallback()
        scheduler.register_callback(cb1)
        scheduler.register_callback(cb2)

        state = GenerationState()
        scheduler._notify_token_generated(torch.tensor([42]), state)

        self.assertEqual(len(cb1.events), 1)
        self.assertEqual(len(cb2.events), 1)

    def test_first_callback_pause_stops_rest(self):
        """When first callback pauses, remaining callbacks are not called."""
        scheduler = GenerationScheduler(mode="force")

        class ImmediatePause(SchedulerCallback):
            def on_token_generated(self, token_id, state, context):
                return False

        cb_after = CustomTestCallback()
        scheduler.register_callback(ImmediatePause())
        scheduler.register_callback(cb_after)

        state = GenerationState()
        result = scheduler._notify_token_generated(torch.tensor([1]), state)

        self.assertFalse(result)
        token_events = [e for e in cb_after.events if e[0] == "token"]
        self.assertEqual(len(token_events), 0)

    def test_logits_callbacks_chain(self):
        """Logits modifications from multiple callbacks are chained."""
        scheduler = GenerationScheduler(mode="force")

        class ScaleCallback(SchedulerCallback):
            def __init__(self, factor):
                self.factor = factor

            def on_logits_ready(self, logits, state, context):
                return logits * self.factor

        scheduler.register_callback(ScaleCallback(2.0))
        scheduler.register_callback(ScaleCallback(3.0))

        logits = torch.ones(1, 5)
        state = GenerationState()
        result = scheduler._notify_logits_ready(logits, state)

        # 1.0 * 2.0 * 3.0 = 6.0
        self.assertTrue(torch.allclose(result, torch.ones(1, 5) * 6.0))


# ==============================================================================
# GenerationConfig Integration Tests
# ==============================================================================


class TestGenerationConfigSchedulerFields(unittest.TestCase):
    """Tests for scheduler fields in GenerationConfig."""

    def test_default_scheduler_fields(self):
        """Default GenerationConfig has scheduler fields as None."""
        from transformers.generation.configuration_utils import GenerationConfig
        config = GenerationConfig()
        self.assertIsNone(config.scheduler_mode)
        self.assertIsNone(config.scheduler_check_interval)
        self.assertIsNone(config.scheduler_step_budget)

    def test_scheduler_fields_from_kwargs(self):
        """Scheduler fields can be set via kwargs."""
        from transformers.generation.configuration_utils import GenerationConfig
        config = GenerationConfig(
            scheduler_mode="force",
            scheduler_check_interval=5,
            scheduler_step_budget=100,
        )
        self.assertEqual(config.scheduler_mode, "force")
        self.assertEqual(config.scheduler_check_interval, 5)
        self.assertEqual(config.scheduler_step_budget, 100)

    def test_scheduler_mode_none_is_default(self):
        """scheduler_mode=None means no scheduler (backward compatible)."""
        from transformers.generation.configuration_utils import GenerationConfig
        config = GenerationConfig()
        # None means not set → no scheduler
        self.assertIsNone(config.scheduler_mode)


# ==============================================================================
# New: add_callback alias tests
# ==============================================================================


class TestAddCallbackAlias(unittest.TestCase):
    """Tests for the add_callback convenience alias."""

    def test_add_callback_works(self):
        """add_callback is an alias for register_callback."""
        scheduler = GenerationScheduler(mode="force")
        cb = SchedulerCallback()
        scheduler.add_callback(cb)
        self.assertIn(cb, scheduler.callbacks)

    def test_add_callback_same_as_register(self):
        """add_callback and register_callback produce same result."""
        s1 = GenerationScheduler(mode="force")
        s2 = GenerationScheduler(mode="force")
        cb = SchedulerCallback()
        s1.register_callback(cb)
        s2.add_callback(cb)
        self.assertEqual(len(s1.callbacks), len(s2.callbacks))


# ==============================================================================
# New: ControlTokenParser.from_tokenizer tests
# ==============================================================================


class TestControlTokenParserFromTokenizer(unittest.TestCase):
    """Tests for ControlTokenParser.from_tokenizer() class method."""

    def test_from_tokenizer_basic(self):
        """from_tokenizer creates a parser with correct token mappings."""
        # Use a mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.additional_special_tokens = []
                self._vocab = {"<|sched:pause|>": 50000, "<|sched:read|>": 50001}
                self.unk_token_id = 0

            def get_vocab(self):
                return dict(self._vocab)

            def add_special_tokens(self, special_tokens_dict):
                new_tokens = special_tokens_dict.get("additional_special_tokens", [])
                for t in new_tokens:
                    if t not in self._vocab:
                        self._vocab[t] = len(self._vocab) + 50000
                self.additional_special_tokens.extend(new_tokens)
                return len(new_tokens)

            def convert_tokens_to_ids(self, token):
                return self._vocab.get(token, self.unk_token_id)

        tokenizer = MockTokenizer()
        parser = ControlTokenParser.from_tokenizer(
            tokenizer,
            control_token_names=["pause", "read"],
            action_handlers={"pause": lambda n, s, c: None},
        )

        self.assertTrue(parser.is_control_token(50000))
        self.assertTrue(parser.is_control_token(50001))
        self.assertEqual(parser.get_action(50000), "pause")
        self.assertEqual(parser.get_action(50001), "read")
        self.assertIn("pause", parser.action_handlers)

    def test_from_tokenizer_custom_prefix_suffix(self):
        """from_tokenizer supports custom prefix and suffix."""
        class MockTokenizer:
            def __init__(self):
                self.additional_special_tokens = []
                self._vocab = {"[CTRL:stop]": 99}
                self.unk_token_id = 0

            def get_vocab(self):
                return dict(self._vocab)

            def add_special_tokens(self, special_tokens_dict):
                new_tokens = special_tokens_dict.get("additional_special_tokens", [])
                for t in new_tokens:
                    if t not in self._vocab:
                        self._vocab[t] = len(self._vocab) + 100
                self.additional_special_tokens.extend(new_tokens)
                return len(new_tokens)

            def convert_tokens_to_ids(self, token):
                return self._vocab.get(token, self.unk_token_id)

        tokenizer = MockTokenizer()
        parser = ControlTokenParser.from_tokenizer(
            tokenizer,
            control_token_names=["stop"],
            prefix="[CTRL:",
            suffix="]",
        )

        self.assertTrue(parser.is_control_token(99))
        self.assertEqual(parser.get_action(99), "stop")

    def test_from_tokenizer_skips_unk(self):
        """from_tokenizer skips tokens that resolve to UNK."""
        class MockTokenizer:
            def __init__(self):
                self.additional_special_tokens = []
                self._vocab = {}
                self.unk_token_id = 0

            def get_vocab(self):
                return dict(self._vocab)

            def add_special_tokens(self, special_tokens_dict):
                return 0  # Pretend we can't add tokens

            def convert_tokens_to_ids(self, token):
                return self.unk_token_id  # Everything is UNK

        tokenizer = MockTokenizer()
        parser = ControlTokenParser.from_tokenizer(
            tokenizer,
            control_token_names=["pause"],
        )

        # No control tokens should have been registered
        self.assertEqual(len(parser.control_tokens), 0)


# ==============================================================================
# New: batch_control_mask tests
# ==============================================================================


class TestBatchControlMask(unittest.TestCase):
    """Tests for the batch_control_mask field in GenerationState."""

    def test_default_is_none(self):
        """batch_control_mask defaults to None."""
        state = GenerationState()
        self.assertIsNone(state.batch_control_mask)

    def test_can_set_mask(self):
        """batch_control_mask can be set to a tensor."""
        mask = torch.ones(4, dtype=torch.long)
        state = GenerationState(batch_control_mask=mask)
        self.assertTrue(torch.equal(state.batch_control_mask, mask))

    def test_clone_preserves_mask(self):
        """Shallow clone preserves batch_control_mask reference."""
        mask = torch.ones(4, dtype=torch.long)
        state = GenerationState(batch_control_mask=mask)
        clone = state.clone(deep_copy_tensors=False)
        self.assertTrue(torch.equal(clone.batch_control_mask, mask))

    def test_deep_clone_copies_mask(self):
        """Deep clone creates independent copy of batch_control_mask."""
        mask = torch.ones(4, dtype=torch.long)
        state = GenerationState(batch_control_mask=mask)
        clone = state.clone(deep_copy_tensors=True)
        clone.batch_control_mask[0] = 0
        self.assertEqual(state.batch_control_mask[0].item(), 1)  # Original unchanged


# ==============================================================================
# New: StreamingSchedulerCallback tests
# ==============================================================================


class TestStreamingSchedulerCallback(unittest.TestCase):
    """Tests for the StreamingSchedulerCallback."""

    def setUp(self):
        from transformers.generation.scheduler_callbacks import StreamingSchedulerCallback
        self.StreamingSchedulerCallback = StreamingSchedulerCallback

    def test_basic_streaming(self):
        """StreamingSchedulerCallback collects text from tokens."""
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "".join([f"t{tid}" for tid in token_ids])

        chunks = []
        cb = self.StreamingSchedulerCallback(
            tokenizer=MockTokenizer(),
            on_text=lambda text: chunks.append(text),
        )

        state = GenerationState(step=0)
        context = SchedulerContext(mode=SchedulerMode.FORCE)

        # Simulate 3 tokens
        for i in range(3):
            state.step = i
            token = torch.tensor([i + 10])
            cb.on_token_generated(token, state, context)

        self.assertGreater(len(chunks), 0)
        self.assertGreater(len(cb.generated_text), 0)

    def test_get_text(self):
        """get_text returns accumulated text."""
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join([str(tid) for tid in token_ids])

        cb = self.StreamingSchedulerCallback(tokenizer=MockTokenizer())
        state = GenerationState(step=0)
        context = SchedulerContext(mode=SchedulerMode.FORCE)

        cb.on_token_generated(torch.tensor([1]), state, context)
        cb.on_token_generated(torch.tensor([2]), state, context)

        text = cb.get_text()
        self.assertIn("1", text)
        self.assertIn("2", text)

    def test_reset(self):
        """reset clears all accumulated state."""
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "text"

        cb = self.StreamingSchedulerCallback(tokenizer=MockTokenizer())
        state = GenerationState(step=0)
        context = SchedulerContext(mode=SchedulerMode.FORCE)

        cb.on_token_generated(torch.tensor([1]), state, context)
        self.assertGreater(len(cb.generated_text), 0)

        cb.reset()
        self.assertEqual(cb.generated_text, "")
        self.assertEqual(len(cb._token_buffer), 0)

    def test_prompt_length_recorded(self):
        """Prompt length is recorded on PREFILL→DECODING transition."""
        class MockTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "text"

        cb = self.StreamingSchedulerCallback(tokenizer=MockTokenizer())
        state = GenerationState(
            phase=GenerationPhase.PREFILL,
            input_ids=torch.randint(0, 100, (1, 20)),
        )
        context = SchedulerContext(mode=SchedulerMode.FORCE)

        cb.on_phase_transition(
            GenerationPhase.PREFILL, GenerationPhase.DECODING, state, context
        )
        self.assertEqual(cb._prompt_length, 20)


if __name__ == "__main__":
    unittest.main()
