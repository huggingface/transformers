"""Strict assistant generation utilities."""

import copy
from typing import Any, Dict, Protocol, Union

from .overlay import AssistantModelProxy, build_overlay_config


class GenerationModel(Protocol):
    """Protocol for models that can generate text."""
    def generate(self, inputs: Dict[str, Any], **kwargs: Any) -> Union[Dict[str, Any], Any]: ...


class ConfiguredModel(Protocol):
    """Protocol for models with generation_config."""
    generation_config: Any


def _extract_assistant_overrides(gen_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract assistant-specific overrides from generation kwargs.

    Pulls out allowed assistant keys to prevent accidental propagation
    to downstream generate() calls.
    """
    allowed_keys = {"num_assistant_tokens", "num_assistant_tokens_schedule"}

    overrides = {}
    for key in allowed_keys:
        if key in gen_kwargs:
            overrides[key] = gen_kwargs.pop(key)

    return overrides


def _snapshot_config(model: ConfiguredModel) -> Dict[str, Any]:
    """Capture a deep snapshot of the model's generation_config for drift detection.

    Creates a comparable copy that's safe for concurrent calls.
    """
    return copy.deepcopy(model.generation_config.to_dict())


class ConfigAccessError(RuntimeError):
    """Raised when assistant config is never accessed during generation."""
    pass


class ConfigDriftError(RuntimeError):
    """Raised when assistant config is modified during generation."""
    pass


def assisted_generate_strict(
    model: GenerationModel,
    inputs: Dict[str, Any],
    assistant_model: ConfiguredModel,
    **gen_kwargs: Any,
) -> Union[Dict[str, Any], Any]:
    """Perform strict assisted generation with overlay protection and drift detection.

    Guarantees assistant overrides are visible via proxy, verifies config access,
    and ensures the real assistant config remains unchanged.
    """
    # Extract and validate assistant overrides
    overrides = _extract_assistant_overrides(gen_kwargs)

    if "num_assistant_tokens" in overrides:
        num_tokens = overrides["num_assistant_tokens"]
        if not isinstance(num_tokens, int) or num_tokens <= 0:
            raise ValueError(
                f"num_assistant_tokens must be a positive integer, got {num_tokens}"
            )

    # TODO: Add validation for num_assistant_tokens_schedule when requirements are clarified

    # Capture baseline config snapshot for drift detection
    pre_call_snapshot = _snapshot_config(assistant_model)

    # Build immutable overlay config and create proxy
    overlay_config = build_overlay_config(assistant_model.generation_config, overrides)
    proxy = AssistantModelProxy(assistant_model, overlay_config)

    # Execute generation with proxied assistant
    result = model.generate(inputs, assistant_model=proxy, **gen_kwargs)

    # Verify config was actually accessed during generation
    if proxy.gen_cfg_reads == 0:
        raise ConfigAccessError(
            "Assistant generation_config was never accessed during the call"
        )

    # Verify no config drift occurred
    post_call_snapshot = _snapshot_config(assistant_model)
    if pre_call_snapshot != post_call_snapshot:
        raise ConfigDriftError(
            "Assistant model configuration was modified during generation"
        )

    return result
