"""Thread-safe overlay utilities for assistant model configurations.

Provides immutable configuration overlays and transparent model proxies
to ensure deterministic behavior during assisted generation under concurrency.
"""

import threading
from typing import Any, Protocol
from weakref import WeakKeyDictionary

from transformers import GenerationConfig


class ModelWithGenerationConfig(Protocol):
    """Protocol for models that have a generation_config attribute."""
    generation_config: GenerationConfig


# Global per-model lock registry to prevent memory leaks
_model_locks: WeakKeyDictionary[object, threading.RLock] = WeakKeyDictionary()


def _lock_for(model: object) -> threading.RLock:
    """Return a per-model reentrant lock using WeakKeyDictionary.

    Each model instance gets its own lock to ensure thread-safe operations
    without creating memory leaks through strong references.
    """
    if model not in _model_locks:
        _model_locks[model] = threading.RLock()
    return _model_locks[model]


class _ImmutableGenerationConfig(GenerationConfig):
    """Immutable wrapper around GenerationConfig.

    Once initialized, prevents any attribute modification to ensure
    no configuration drift during concurrent assisted generation calls.
    Allows safe mutations of token IDs for Hugging Face internals.
    """

    _frozen: bool
    SAFE_MUTABLE = {"eos_token_id", "pad_token_id", "bos_token_id"}

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the config and freeze it."""
        super().__init__(**kwargs)  # type: ignore[misc]
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification after initialization except for safe mutable attributes."""
        if hasattr(self, '_frozen') and self._frozen:
            # Allow Hugging Face internals to modify safe token IDs
            if name in self.SAFE_MUTABLE:
                super().__setattr__(name, value)
                return

            raise AttributeError(
                f"Cannot modify frozen GenerationConfig attribute '{name}'"
            )
        super().__setattr__(name, value)


def build_overlay_config(
    base: GenerationConfig, overrides: dict[str, Any]
) -> _ImmutableGenerationConfig:
    """Build an immutable config by merging base with overrides.

    Creates a new immutable configuration from the base config,
    applying only the provided overrides (ignoring None values).
    """
    config_dict = base.to_dict()

    for key, value in overrides.items():
        if value is not None:
            config_dict[key] = value

    return _ImmutableGenerationConfig(**config_dict)


class AssistantModelProxy:
    """Transparent proxy for assistant models with immutable generation_config.

    Wraps an assistant model to provide read-only access to an overlay
    configuration while tracking access counts and delegating all other
    operations to the wrapped model.
    """

    _wrapped: ModelWithGenerationConfig
    _overlay_cfg: GenerationConfig
    _gen_cfg_reads: int

    def __init__(self, wrapped: ModelWithGenerationConfig, overlay_cfg: GenerationConfig) -> None:
        """Initialize proxy with wrapped model and overlay config."""
        object.__setattr__(self, '_wrapped', wrapped)
        object.__setattr__(self, '_overlay_cfg', overlay_cfg)
        object.__setattr__(self, '_gen_cfg_reads', 0)

    @property
    def generation_config(self) -> GenerationConfig:
        """Get overlay configuration and increment access counter."""
        # Use per-model lock instead of global to avoid blocking unrelated models
        with _lock_for(self._wrapped):
            object.__setattr__(self, '_gen_cfg_reads', self._gen_cfg_reads + 1)
        return self._overlay_cfg

    @property
    def gen_cfg_reads(self) -> int:
        """Number of times generation_config was accessed."""
        return self._gen_cfg_reads

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate attribute assignment while protecting generation_config."""
        if name == 'generation_config':
            raise AttributeError(
                "Cannot reassign generation_config on AssistantModelProxy"
            )
        setattr(self._wrapped, name, value)

    def __repr__(self) -> str:
        """Return debug representation showing wrapped model and config reads."""
        return (
            f"AssistantModelProxy(wrapped={self._wrapped!r}, "
            f"reads={self._gen_cfg_reads})"
        )

    def __str__(self) -> str:
        """Return string representation delegated to wrapped model."""
        return str(self._wrapped)
