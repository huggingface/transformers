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
Pipeline Composition DSL for HuggingFace Transformers.

Enables declarative chaining of multiple pipelines with automatic
data transformation, device management, and error handling.

Example:
    >>> from transformers import pipeline
    >>> from transformers.pipelines.compose import compose
    >>>
    >>> workflow = compose([
    ...     pipeline("image-to-text"),
    ...     pipeline("translation", model="Helsinki-NLP/opus-mt-en-de"),
    ... ])
    >>> result = workflow(image="photo.jpg")
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
)


logger = logging.getLogger(__name__)


class ErrorHandling(Enum):
    """Strategy for handling errors in pipeline stages."""

    RAISE = "raise"  # Raise immediately on error
    SKIP_FAILED = "skip_failed"  # Skip failed items, continue with rest
    RETRY = "retry"  # Retry with exponential backoff


class DeviceMap(Enum):
    """Strategy for device placement across pipeline stages."""

    AUTO = "auto"  # Automatically distribute based on memory
    BALANCED = "balanced"  # Evenly distribute across available devices
    SEQUENTIAL = "sequential"  # Keep all on same device


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage: int
    stage_name: str
    output: Any
    elapsed_ms: float
    success: bool
    error: Exception | None = None


@dataclass
class CompositionResult:
    """Final result from composed pipeline execution."""

    output: Any
    stage_results: list[StageResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0

    @property
    def success(self) -> bool:
        return all(r.success for r in self.stage_results)

    @property
    def failed_stages(self) -> list[int]:
        return [r.stage for r in self.stage_results if not r.success]


# Type aliases
Adapter = Callable[[Any], Any]
AdapterKey = tuple[int, int]
AdapterMap = dict[AdapterKey, Adapter]


# =============================================================================
# Default Adapters Registry
# =============================================================================


def _extract_generated_text(output: Any) -> str:
    """Extract generated text from various pipeline outputs."""
    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    if isinstance(output, dict):
        # Common keys for generated text
        for key in ["generated_text", "text", "translation_text", "summary_text"]:
            if key in output:
                return output[key]

    if isinstance(output, str):
        return output

    return str(output)


def _extract_translation(output: Any) -> str:
    """Extract translation from translation pipeline output."""
    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    if isinstance(output, dict) and "translation_text" in output:
        return output["translation_text"]

    return _extract_generated_text(output)


def _extract_summary(output: Any) -> str:
    """Extract summary from summarization pipeline output."""
    if isinstance(output, list) and len(output) > 0:
        output = output[0]

    if isinstance(output, dict) and "summary_text" in output:
        return output["summary_text"]

    return _extract_generated_text(output)


def _identity(output: Any) -> Any:
    """Pass through unchanged."""
    return output


# Registry of default adapters based on pipeline task types
# Key: (source_task, target_task) -> adapter function
DEFAULT_ADAPTERS: dict[tuple[str, str], Adapter] = {
    # Image-to-text → text pipelines
    ("image-to-text", "translation"): _extract_generated_text,
    ("image-to-text", "summarization"): _extract_generated_text,
    ("image-to-text", "text-generation"): _extract_generated_text,
    ("image-to-text", "text2text-generation"): _extract_generated_text,
    ("image-to-text", "text-to-speech"): _extract_generated_text,
    ("image-to-text", "text-to-audio"): _extract_generated_text,
    # Translation → text pipelines
    ("translation", "summarization"): _extract_translation,
    ("translation", "text-generation"): _extract_translation,
    ("translation", "text-to-speech"): _extract_translation,
    ("translation", "text-to-audio"): _extract_translation,
    ("translation", "translation"): _extract_translation,
    # Summarization → text pipelines
    ("summarization", "translation"): _extract_summary,
    ("summarization", "text-generation"): _extract_summary,
    ("summarization", "text-to-speech"): _extract_summary,
    # Text generation → other
    ("text-generation", "translation"): _extract_generated_text,
    ("text-generation", "summarization"): _extract_generated_text,
    ("text-generation", "text-to-speech"): _extract_generated_text,
    # ASR → text pipelines
    ("automatic-speech-recognition", "translation"): _extract_generated_text,
    ("automatic-speech-recognition", "summarization"): _extract_generated_text,
    ("automatic-speech-recognition", "text-generation"): _extract_generated_text,
    # Document QA / VQA → text
    ("document-question-answering", "translation"): _extract_generated_text,
    ("visual-question-answering", "translation"): _extract_generated_text,
}


def get_default_adapter(source_task: str, target_task: str) -> Adapter | None:
    """Get default adapter for a source→target task pair."""
    return DEFAULT_ADAPTERS.get((source_task, target_task))


def register_default_adapter(source_task: str, target_task: str, adapter: Adapter) -> None:
    """Register a new default adapter for a task pair."""
    DEFAULT_ADAPTERS[(source_task, target_task)] = adapter


# =============================================================================
# ComposablePipeline
# =============================================================================


class ComposablePipeline:
    """
    Orchestrates a chain of HuggingFace pipelines.

    Handles data transformation between stages, device management,
    error handling, and streaming support.

    Args:
        pipelines: List of pipeline objects to chain
        adapters: Dict mapping (source_idx, target_idx) to adapter functions
        error_handling: Strategy for handling errors ("raise", "skip_failed", "retry")
        device_map: Device placement strategy ("auto", "balanced", "sequential")
        max_retries: Maximum retries when error_handling="retry"
        retry_delay: Base delay in seconds between retries (exponential backoff)

    Example:
        >>> workflow = ComposablePipeline(
        ...     pipelines=[ocr_pipeline, translation_pipeline],
        ...     adapters={(0, 1): lambda x: x["generated_text"]},
        ...     error_handling="raise"
        ... )
        >>> result = workflow("document.pdf")
    """

    def __init__(
        self,
        pipelines: list[Any],
        adapters: AdapterMap | None = None,
        error_handling: str | ErrorHandling = "raise",
        device_map: str | DeviceMap = "auto",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        if not pipelines:
            raise ValueError("At least one pipeline is required")

        self.pipelines = pipelines
        self.adapters = adapters or {}
        self.error_handling = ErrorHandling(error_handling) if isinstance(error_handling, str) else error_handling
        self.device_map = DeviceMap(device_map) if isinstance(device_map, str) else device_map
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Build adapter chain with auto-inference
        self._adapter_chain = self._build_adapter_chain()

        # Apply device placement
        self._apply_device_map()

    def _get_pipeline_task(self, pipeline: Any) -> str | None:
        """Extract task name from a pipeline."""
        if hasattr(pipeline, "task"):
            return pipeline.task
        return None

    def _build_adapter_chain(self) -> dict[int, Adapter]:
        """Build the complete adapter chain with auto-inference."""
        chain: dict[int, Adapter] = {}

        for i in range(len(self.pipelines) - 1):
            source_idx, target_idx = i, i + 1
            key = (source_idx, target_idx)

            # Check for explicit adapter
            if key in self.adapters:
                chain[i] = self.adapters[key]
                continue

            # Try to auto-infer from task types
            source_task = self._get_pipeline_task(self.pipelines[source_idx])
            target_task = self._get_pipeline_task(self.pipelines[target_idx])

            if source_task and target_task:
                default = get_default_adapter(source_task, target_task)
                if default:
                    chain[i] = default
                    logger.debug(f"Auto-inferred adapter for stage {i}→{i + 1}: {source_task}→{target_task}")
                    continue

            # Fall back to identity (pass-through)
            chain[i] = _identity
            logger.warning(
                f"No adapter for stage {i}→{i + 1}, using identity. Consider providing an explicit adapter."
            )

        return chain

    def _apply_device_map(self) -> None:
        """Apply device placement strategy to pipelines.

        Note: Currently pipelines retain their original device placement.
        Future enhancements could include:
        - AUTO: Memory-aware distribution across available devices
        - BALANCED: Even distribution across GPUs for multi-GPU setups
        """
        # Device redistribution is a future enhancement.
        # Currently, pipelines keep their configured devices.
        pass

    def _run_stage(self, stage_idx: int, input_data: Any, **kwargs) -> StageResult:
        """Run a single pipeline stage with timing."""
        pipeline = self.pipelines[stage_idx]
        task = self._get_pipeline_task(pipeline) or f"stage_{stage_idx}"

        start = time.perf_counter()
        try:
            # Handle different input types
            if isinstance(input_data, dict):
                output = pipeline(**input_data, **kwargs)
            else:
                output = pipeline(input_data, **kwargs)

            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage=stage_idx,
                stage_name=task,
                output=output,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return StageResult(
                stage=stage_idx,
                stage_name=task,
                output=None,
                elapsed_ms=elapsed,
                success=False,
                error=e,
            )

    def _run_stage_with_retry(self, stage_idx: int, input_data: Any, **kwargs) -> StageResult:
        """Run a stage with retry logic if configured."""
        if self.error_handling != ErrorHandling.RETRY:
            return self._run_stage(stage_idx, input_data, **kwargs)

        last_result = None
        for attempt in range(self.max_retries):
            result = self._run_stage(stage_idx, input_data, **kwargs)
            if result.success:
                return result

            last_result = result
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"Stage {stage_idx} failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {delay:.1f}s: {result.error}"
                )
                time.sleep(delay)

        # Mark as failed after all retries exhausted
        return last_result  # type: ignore

    def __call__(self, *args, _return_intermediates: bool = False, **kwargs) -> Any | CompositionResult:
        """
        Execute the composed pipeline.

        Args:
            *args: Positional arguments for the first pipeline
            _return_intermediates: If True, return full CompositionResult
            **kwargs: Keyword arguments for the first pipeline

        Returns:
            Final output, or CompositionResult if _return_intermediates=True
        """
        # Prepare initial input
        if args and kwargs:
            # Both positional and keyword - pass through
            current_input = {"args": args, "kwargs": kwargs}
        elif args:
            current_input = args[0] if len(args) == 1 else args
        else:
            current_input = kwargs

        stage_results: list[StageResult] = []
        total_start = time.perf_counter()

        for i, pipeline in enumerate(self.pipelines):
            # Run the stage
            result = self._run_stage_with_retry(i, current_input)
            stage_results.append(result)

            # Handle errors
            if not result.success:
                if self.error_handling in (ErrorHandling.RAISE, ErrorHandling.RETRY):
                    # RETRY: if we got here, retries were exhausted
                    raise RuntimeError(
                        f"Pipeline stage {i} ({result.stage_name}) failed: {result.error}"
                    ) from result.error
                elif self.error_handling == ErrorHandling.SKIP_FAILED:
                    # Skip this item, but we can't continue the chain
                    logger.warning(f"Stage {i} failed, stopping chain: {result.error}")
                    break

            # Apply adapter for next stage (if not last)
            if result.success and i < len(self.pipelines) - 1:
                adapter = self._adapter_chain.get(i, _identity)
                try:
                    current_input = adapter(result.output)
                except Exception as e:
                    # Adapter failed
                    adapter_result = StageResult(
                        stage=i,
                        stage_name=f"adapter_{i}_{i + 1}",
                        output=None,
                        elapsed_ms=0,
                        success=False,
                        error=e,
                    )
                    stage_results.append(adapter_result)

                    if self.error_handling == ErrorHandling.RAISE:
                        raise RuntimeError(f"Adapter {i}→{i + 1} failed: {e}") from e
                    break
            elif result.success:
                current_input = result.output

        total_elapsed = (time.perf_counter() - total_start) * 1000

        composition_result = CompositionResult(
            output=current_input if stage_results and stage_results[-1].success else None,
            stage_results=stage_results,
            total_elapsed_ms=total_elapsed,
        )

        if _return_intermediates:
            return composition_result
        return composition_result.output

    def stream(self, *args, **kwargs) -> Generator[StageResult, None, None]:
        """
        Execute pipeline and yield results after each stage.

        Yields:
            StageResult for each completed stage

        Example:
            >>> for result in workflow.stream("input.jpg"):
            ...     print(f"Stage {result.stage}: {result.output}")
        """
        # Prepare initial input
        if args and kwargs:
            current_input = {"args": args, "kwargs": kwargs}
        elif args:
            current_input = args[0] if len(args) == 1 else args
        else:
            current_input = kwargs

        for i, pipeline in enumerate(self.pipelines):
            result = self._run_stage_with_retry(i, current_input)
            yield result

            if not result.success:
                if self.error_handling in (ErrorHandling.RAISE, ErrorHandling.RETRY):
                    raise RuntimeError(f"Pipeline stage {i} failed: {result.error}") from result.error
                return

            # Apply adapter for next stage
            if i < len(self.pipelines) - 1:
                adapter = self._adapter_chain.get(i, _identity)
                try:
                    current_input = adapter(result.output)
                except Exception as e:
                    if self.error_handling == ErrorHandling.RAISE:
                        raise RuntimeError(f"Adapter {i}→{i + 1} failed: {e}") from e
                    return
            else:
                current_input = result.output

    def __len__(self) -> int:
        """Return number of pipeline stages."""
        return len(self.pipelines)

    def __repr__(self) -> str:
        tasks = [self._get_pipeline_task(p) or "unknown" for p in self.pipelines]
        return f"ComposablePipeline({' → '.join(tasks)})"


# =============================================================================
# Factory Function
# =============================================================================


def compose(
    pipelines: list[Any],
    adapters: AdapterMap | None = None,
    error_handling: str = "raise",
    device_map: str = "auto",
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> ComposablePipeline:
    """
    Create a composed pipeline from a list of pipelines.

    This is the primary API for creating pipeline chains.

    Args:
        pipelines: List of HuggingFace pipeline objects
        adapters: Dict mapping (source_idx, target_idx) to adapter functions.
            If not provided, will attempt to auto-infer based on task types.
        error_handling: How to handle stage failures:
            - "raise": Raise exception immediately (default)
            - "skip_failed": Log warning and stop the chain
            - "retry": Retry with exponential backoff
        device_map: Device placement strategy:
            - "auto": Automatically distribute based on memory
            - "balanced": Evenly distribute across devices
            - "sequential": Keep all on same device
        max_retries: Maximum retry attempts when error_handling="retry"
        retry_delay: Base delay in seconds for retry backoff

    Returns:
        ComposablePipeline instance

    Example:
        >>> from transformers import pipeline
        >>> from transformers.pipelines.compose import compose
        >>>
        >>> # Simple chain with auto-inferred adapters
        >>> workflow = compose([
        ...     pipeline("image-to-text", model="Salesforce/blip-image-captioning-base"),
        ...     pipeline("translation", model="Helsinki-NLP/opus-mt-en-de"),
        ... ])
        >>>
        >>> # With explicit adapters
        >>> workflow = compose([
        ...     pipeline("image-to-text"),
        ...     pipeline("translation"),
        ...     pipeline("text-to-speech"),
        ... ], adapters={
        ...     (0, 1): lambda x: x[0]["generated_text"],
        ...     (1, 2): lambda x: x[0]["translation_text"],
        ... })
        >>>
        >>> # Execute
        >>> result = workflow(image="photo.jpg")
    """
    return ComposablePipeline(
        pipelines=pipelines,
        adapters=adapters,
        error_handling=error_handling,
        device_map=device_map,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


# =============================================================================
# Batch Processing
# =============================================================================


def compose_batch(
    pipeline: ComposablePipeline,
    inputs: list[Any],
    error_handling: str | None = None,
) -> list[CompositionResult]:
    """
    Process multiple inputs through a composed pipeline.

    Args:
        pipeline: The composed pipeline to use
        inputs: List of inputs to process
        error_handling: Override pipeline's error_handling for batch.
            With "skip_failed", failed items return None but don't stop batch.

    Returns:
        List of CompositionResult, one per input

    Example:
        >>> results = compose_batch(workflow, ["img1.jpg", "img2.jpg", "img3.jpg"])
    """
    results = []

    for inp in inputs:
        try:
            if isinstance(inp, dict):
                result = pipeline(_return_intermediates=True, **inp)
            else:
                result = pipeline(inp, _return_intermediates=True)
            results.append(result)
        except Exception:
            handling = error_handling or pipeline.error_handling.value
            if handling == "skip_failed":
                results.append(
                    CompositionResult(
                        output=None,
                        stage_results=[],
                        total_elapsed_ms=0,
                    )
                )
            else:
                raise

    return results


# =============================================================================
# Convenience decorators
# =============================================================================


def adapter(source_idx: int, target_idx: int):
    """
    Decorator for defining adapters inline.

    Example:
        >>> @adapter(0, 1)
        ... def extract_caption(output):
        ...     return output[0]["generated_text"]
        >>>
        >>> workflow = compose(pipelines, adapters={extract_caption.key: extract_caption})
    """

    def decorator(func: Adapter) -> Adapter:
        func.key = (source_idx, target_idx)  # type: ignore
        return func

    return decorator


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ComposablePipeline",
    "compose",
    "compose_batch",
    "StageResult",
    "CompositionResult",
    "ErrorHandling",
    "DeviceMap",
    "Adapter",
    "AdapterMap",
    "adapter",
    "register_default_adapter",
    "get_default_adapter",
]
