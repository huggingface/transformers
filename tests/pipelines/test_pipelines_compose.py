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
Comprehensive tests for Pipeline Composition DSL.

Tests cover:
- Basic pipeline chaining
- Adapter system (explicit and auto-inferred)
- Error handling strategies
- Streaming support
- Batch processing
- Edge cases
"""

from typing import Any

import pytest

from transformers.pipelines.compose import (
    ComposablePipeline,
    CompositionResult,
    DeviceMap,
    StageResult,
    _extract_generated_text,
    _extract_translation,
    _identity,
    adapter,
    compose,
    compose_batch,
    get_default_adapter,
    register_default_adapter,
)


# =============================================================================
# Mock Pipelines for Testing
# =============================================================================


class MockPipeline:
    """Mock pipeline for testing without loading real models."""

    def __init__(
        self,
        task: str,
        output: Any = None,
        should_fail: bool = False,
        fail_count: int = 0,
    ):
        self.task = task
        self._output = output
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._call_count = 0
        self._inputs: list[Any] = []

    def __call__(self, *args, **kwargs) -> Any:
        self._call_count += 1
        self._inputs.append((args, kwargs))

        # Support transient failures for retry testing
        if self._fail_count > 0 and self._call_count <= self._fail_count:
            raise RuntimeError(f"Mock failure {self._call_count}/{self._fail_count}")

        if self._should_fail:
            raise RuntimeError(f"Mock pipeline '{self.task}' failed")

        return self._output

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_input(self) -> Any:
        return self._inputs[-1] if self._inputs else None


def make_image_to_text_pipeline(output_text: str = "A photo of a cat") -> MockPipeline:
    """Create mock image-to-text pipeline."""
    return MockPipeline(task="image-to-text", output=[{"generated_text": output_text}])


def make_translation_pipeline(
    output_text: str = "Ein Foto einer Katze",
) -> MockPipeline:
    """Create mock translation pipeline."""
    return MockPipeline(task="translation", output=[{"translation_text": output_text}])


def make_summarization_pipeline(output_text: str = "Summary of text") -> MockPipeline:
    """Create mock summarization pipeline."""
    return MockPipeline(task="summarization", output=[{"summary_text": output_text}])


def make_tts_pipeline(output: Any = None) -> MockPipeline:
    """Create mock text-to-speech pipeline."""
    if output is None:
        output = {"audio": b"fake_audio"}
    return MockPipeline(task="text-to-speech", output=output)


# =============================================================================
# Basic Composition Tests
# =============================================================================


class TestBasicComposition:
    """Test basic pipeline composition functionality."""

    def test_compose_creates_composable_pipeline(self):
        """compose() returns a ComposablePipeline."""
        p1 = make_image_to_text_pipeline()
        p2 = make_translation_pipeline()

        workflow = compose([p1, p2])

        assert isinstance(workflow, ComposablePipeline)
        assert len(workflow) == 2

    def test_compose_requires_at_least_one_pipeline(self):
        """compose() raises ValueError for empty list."""
        with pytest.raises(ValueError, match="At least one pipeline"):
            compose([])

    def test_simple_two_stage_chain(self):
        """Two pipelines chain correctly with explicit adapter."""
        p1 = make_image_to_text_pipeline("Hello world")
        p2 = make_translation_pipeline("Hallo Welt")

        workflow = compose([p1, p2], adapters={(0, 1): lambda x: x[0]["generated_text"]})

        result = workflow("image.jpg")

        assert result == [{"translation_text": "Hallo Welt"}]
        assert p1.call_count == 1
        assert p2.call_count == 1
        # Check p2 received adapted input
        assert p2.last_input[0][0] == "Hello world"

    def test_three_stage_chain(self):
        """Three pipelines chain correctly."""
        p1 = make_image_to_text_pipeline("Original caption")
        p2 = make_translation_pipeline("Translated caption")
        p3 = make_tts_pipeline({"audio": b"audio_data"})

        workflow = compose(
            [p1, p2, p3],
            adapters={
                (0, 1): lambda x: x[0]["generated_text"],
                (1, 2): lambda x: x[0]["translation_text"],
            },
        )

        result = workflow("image.jpg")

        assert result == {"audio": b"audio_data"}
        assert p1.call_count == 1
        assert p2.call_count == 1
        assert p3.call_count == 1

    def test_repr(self):
        """ComposablePipeline has useful repr."""
        p1 = make_image_to_text_pipeline()
        p2 = make_translation_pipeline()

        workflow = compose([p1, p2])

        assert "image-to-text" in repr(workflow)
        assert "translation" in repr(workflow)
        assert "→" in repr(workflow)


# =============================================================================
# Adapter System Tests
# =============================================================================


class TestAdapterSystem:
    """Test adapter inference and custom adapters."""

    def test_auto_inferred_adapter_image_to_translation(self):
        """Auto-infers adapter for image-to-text → translation."""
        p1 = make_image_to_text_pipeline("Caption text")
        p2 = make_translation_pipeline("Übersetzung")

        # No explicit adapters
        workflow = compose([p1, p2])
        workflow("image.jpg")  # Execute to trigger adapter

        # Should auto-extract generated_text
        assert p2.last_input[0][0] == "Caption text"

    def test_auto_inferred_adapter_translation_to_summarization(self):
        """Auto-infers adapter for translation → summarization."""
        p1 = make_translation_pipeline("Translated text here")
        p2 = make_summarization_pipeline("Summary")

        workflow = compose([p1, p2])
        workflow("input text")  # Execute to trigger adapter

        assert p2.last_input[0][0] == "Translated text here"

    def test_explicit_adapter_overrides_default(self):
        """Explicit adapter takes precedence over auto-inferred."""
        p1 = make_image_to_text_pipeline("Auto text")
        p2 = make_translation_pipeline("Result")

        def custom_adapter(x):
            return "CUSTOM_OUTPUT"

        workflow = compose([p1, p2], adapters={(0, 1): custom_adapter})
        workflow("image.jpg")  # Execute to trigger adapter

        assert p2.last_input[0][0] == "CUSTOM_OUTPUT"

    def test_identity_adapter_fallback(self):
        """Falls back to identity when no adapter found."""
        # Two pipelines with no default adapter
        p1 = MockPipeline(task="custom-task-1", output={"data": 123})
        p2 = MockPipeline(task="custom-task-2", output="final")

        workflow = compose([p1, p2])
        workflow("input")  # Execute to trigger adapter

        # p2 should receive p1's output unchanged (passed as kwargs since it's a dict)
        assert p2.last_input[1] == {"data": 123}

    def test_register_custom_default_adapter(self):
        """Can register new default adapters."""

        def custom_adapter(x):
            return x["custom_field"]

        register_default_adapter("task-a", "task-b", custom_adapter)

        assert get_default_adapter("task-a", "task-b") == custom_adapter

    def test_adapter_decorator(self):
        """@adapter decorator attaches key to function."""

        @adapter(0, 1)
        def my_adapter(x):
            return x["text"]

        assert hasattr(my_adapter, "key")
        assert my_adapter.key == (0, 1)


# =============================================================================
# Extraction Function Tests
# =============================================================================


class TestExtractionFunctions:
    """Test the built-in extraction helper functions."""

    def test_extract_generated_text_from_dict(self):
        """Extracts generated_text from dict."""
        output = {"generated_text": "Hello"}
        assert _extract_generated_text(output) == "Hello"

    def test_extract_generated_text_from_list(self):
        """Extracts from list of dicts."""
        output = [{"generated_text": "Hello"}]
        assert _extract_generated_text(output) == "Hello"

    def test_extract_generated_text_fallback(self):
        """Falls back to text, translation_text, etc."""
        assert _extract_generated_text({"text": "A"}) == "A"
        assert _extract_generated_text({"translation_text": "B"}) == "B"
        assert _extract_generated_text({"summary_text": "C"}) == "C"

    def test_extract_generated_text_string(self):
        """Returns strings unchanged."""
        assert _extract_generated_text("plain string") == "plain string"

    def test_extract_translation(self):
        """Extracts translation_text specifically."""
        output = [{"translation_text": "Hallo"}]
        assert _extract_translation(output) == "Hallo"

    def test_identity(self):
        """Identity adapter passes through unchanged."""
        obj = {"any": "object"}
        assert _identity(obj) is obj


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling strategies."""

    def test_raise_strategy_propagates_error(self):
        """RAISE strategy re-raises pipeline errors."""
        p1 = make_image_to_text_pipeline()
        p2 = MockPipeline(task="translation", should_fail=True)

        workflow = compose([p1, p2], error_handling="raise")

        with pytest.raises(RuntimeError, match="stage 1.*failed"):
            workflow("image.jpg")

    def test_skip_failed_stops_chain(self):
        """SKIP_FAILED logs warning and stops the chain."""
        p1 = make_image_to_text_pipeline()
        p2 = MockPipeline(task="translation", should_fail=True)
        p3 = make_tts_pipeline()

        workflow = compose([p1, p2, p3], error_handling="skip_failed")

        result = workflow("image.jpg", _return_intermediates=True)

        assert not result.success
        assert 1 in result.failed_stages
        assert p3.call_count == 0  # Never reached

    def test_retry_strategy_retries_on_failure(self):
        """RETRY strategy retries failed stages."""
        p1 = make_image_to_text_pipeline()
        # Fails first 2 times, succeeds on 3rd
        p2 = MockPipeline(task="translation", output="success", fail_count=2)

        workflow = compose(
            [p1, p2],
            error_handling="retry",
            max_retries=3,
            retry_delay=0.01,  # Fast for testing
        )

        result = workflow("image.jpg")

        assert result == "success"
        assert p2.call_count == 3

    def test_retry_exhausted_raises(self):
        """RETRY raises after max retries exhausted."""
        p1 = make_image_to_text_pipeline()
        p2 = MockPipeline(task="translation", should_fail=True)

        workflow = compose(
            [p1, p2],
            error_handling="retry",
            max_retries=2,
            retry_delay=0.01,
        )

        with pytest.raises(RuntimeError):
            workflow("image.jpg")

        assert p2.call_count == 2

    def test_adapter_failure_raises(self):
        """Adapter failure raises with RAISE strategy."""
        p1 = make_image_to_text_pipeline()
        p2 = make_translation_pipeline()

        def bad_adapter(x):
            raise ValueError("Adapter error")

        workflow = compose(
            [p1, p2],
            adapters={(0, 1): bad_adapter},
            error_handling="raise",
        )

        with pytest.raises(RuntimeError, match="Adapter.*failed"):
            workflow("image.jpg")


# =============================================================================
# Streaming Tests
# =============================================================================


class TestStreaming:
    """Test streaming/generator support."""

    def test_stream_yields_stage_results(self):
        """stream() yields StageResult after each stage."""
        p1 = make_image_to_text_pipeline("Caption")
        p2 = make_translation_pipeline("Übersetzung")

        workflow = compose([p1, p2], adapters={(0, 1): lambda x: x[0]["generated_text"]})

        results = list(workflow.stream("image.jpg"))

        assert len(results) == 2
        assert all(isinstance(r, StageResult) for r in results)
        assert results[0].stage == 0
        assert results[0].stage_name == "image-to-text"
        assert results[1].stage == 1
        assert results[1].stage_name == "translation"

    def test_stream_includes_timing(self):
        """StageResult includes elapsed time."""
        p1 = make_image_to_text_pipeline()

        workflow = compose([p1])

        results = list(workflow.stream("input"))

        assert results[0].elapsed_ms >= 0

    def test_stream_stops_on_error_with_raise(self):
        """stream() stops and raises on error with RAISE."""
        p1 = make_image_to_text_pipeline()
        p2 = MockPipeline(task="translation", should_fail=True)

        workflow = compose([p1, p2], error_handling="raise")

        gen = workflow.stream("image.jpg")
        result1 = next(gen)  # First stage OK
        assert result1.success

        # Second stage yields failed result first
        result2 = next(gen)
        assert not result2.success

        # Then raises on next iteration
        with pytest.raises(RuntimeError):
            next(gen)

    def test_stream_stops_on_error_with_skip(self):
        """stream() stops gracefully with SKIP_FAILED."""
        p1 = make_image_to_text_pipeline()
        p2 = MockPipeline(task="translation", should_fail=True)
        p3 = make_tts_pipeline()

        workflow = compose([p1, p2, p3], error_handling="skip_failed")

        results = list(workflow.stream("image.jpg"))

        # Yields 2 results (p1 success, p2 fail), then stops
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Test batch processing functionality."""

    def test_compose_batch_processes_multiple_inputs(self):
        """compose_batch processes list of inputs."""
        p1 = MockPipeline(task="text-gen", output="output")

        workflow = compose([p1])

        results = compose_batch(workflow, ["a", "b", "c"])

        assert len(results) == 3
        assert all(isinstance(r, CompositionResult) for r in results)
        assert p1.call_count == 3

    def test_compose_batch_skip_failed(self):
        """compose_batch with skip_failed continues on errors."""
        call_count = [0]

        class FlakyPipeline:
            task = "flaky"

            def __call__(self, x):
                call_count[0] += 1
                if x == "bad":
                    raise RuntimeError("Bad input")
                return f"processed_{x}"

        workflow = compose([FlakyPipeline()], error_handling="raise")

        results = compose_batch(workflow, ["good1", "bad", "good2"], error_handling="skip_failed")

        assert len(results) == 3
        assert results[0].output == "processed_good1"
        assert results[1].output is None  # Failed
        assert results[2].output == "processed_good2"


# =============================================================================
# CompositionResult Tests
# =============================================================================


class TestCompositionResult:
    """Test CompositionResult dataclass."""

    def test_success_property_all_passed(self):
        """success is True when all stages pass."""
        result = CompositionResult(
            output="final",
            stage_results=[
                StageResult(0, "a", "out", 10, True),
                StageResult(1, "b", "out", 10, True),
            ],
        )
        assert result.success

    def test_success_property_some_failed(self):
        """success is False when any stage failed."""
        result = CompositionResult(
            output=None,
            stage_results=[
                StageResult(0, "a", "out", 10, True),
                StageResult(1, "b", None, 10, False, RuntimeError()),
            ],
        )
        assert not result.success

    def test_failed_stages_property(self):
        """failed_stages lists indices of failed stages."""
        result = CompositionResult(
            output=None,
            stage_results=[
                StageResult(0, "a", "out", 10, True),
                StageResult(1, "b", None, 10, False),
                StageResult(2, "c", "out", 10, True),
                StageResult(3, "d", None, 10, False),
            ],
        )
        assert result.failed_stages == [1, 3]


# =============================================================================
# Input Handling Tests
# =============================================================================


class TestInputHandling:
    """Test various input formats."""

    def test_positional_input(self):
        """Handles positional argument."""
        p1 = MockPipeline(task="test", output="done")
        workflow = compose([p1])

        workflow("input_string")  # Execute pipeline

        assert p1.last_input[0][0] == "input_string"

    def test_keyword_input(self):
        """Handles keyword arguments."""
        p1 = MockPipeline(task="test", output="done")
        workflow = compose([p1])

        workflow(image="photo.jpg", question="What is this?")  # Execute pipeline

        assert p1.last_input[1] == {"image": "photo.jpg", "question": "What is this?"}

    def test_return_intermediates(self):
        """_return_intermediates=True returns full result."""
        p1 = make_image_to_text_pipeline()
        workflow = compose([p1])

        result = workflow("input", _return_intermediates=True)

        assert isinstance(result, CompositionResult)
        assert len(result.stage_results) == 1
        assert result.total_elapsed_ms > 0


# =============================================================================
# Device Map Tests
# =============================================================================


class TestDeviceMap:
    """Test device placement strategies."""

    def test_device_map_enum_values(self):
        """DeviceMap has expected values."""
        assert DeviceMap.AUTO.value == "auto"
        assert DeviceMap.BALANCED.value == "balanced"
        assert DeviceMap.SEQUENTIAL.value == "sequential"

    def test_compose_accepts_device_map_string(self):
        """compose() accepts device_map as string."""
        p1 = make_image_to_text_pipeline()

        workflow = compose([p1], device_map="balanced")

        assert workflow.device_map == DeviceMap.BALANCED

    def test_compose_accepts_device_map_enum(self):
        """compose() accepts device_map as DeviceMap."""
        p1 = make_image_to_text_pipeline()

        workflow = compose([p1], device_map=DeviceMap.SEQUENTIAL)

        assert workflow.device_map == DeviceMap.SEQUENTIAL
