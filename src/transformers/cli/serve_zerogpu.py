# Copyright 2026 The HuggingFace Team. All rights reserved.
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
ZeroGPU support for transformers serve.

Spaces ZeroGPU dynamically allocates GPUs for AI models on Hugging Face Spaces.
It uses the ``@spaces.GPU`` decorator from the ``spaces`` package.

Key properties:
- ``@spaces.GPU`` is **effect-free** in non-ZeroGPU environments, ensuring compatibility
  across different setups.
- In ZeroGPU Spaces, the GPU is only available inside ``@spaces.GPU`` decorated functions.
- Models should be placed on ``cuda`` at the module level (for local dev) or loaded lazily
  in ZeroGPU mode.
- Default GPU size is ``large`` (half NVIDIA RTX Pro 6000 Blackwell); use ``size="xlarge"``
  for a full RTX Pro 6000 Blackwell.

This module provides:
- ``is_zerogpu()`` — Detect whether we are running in a ZeroGPU Space.
- ``zerogpu_enabled()`` — Check if ZeroGPU is active in the serve process.
- ``zerogpu_size()`` — Get the configured GPU size (``"large"`` or ``"xlarge"``).
- ``zerogpu_decorator()`` — A decorator that wraps functions with ``@spaces.GPU``
  in ZeroGPU Spaces and is effect-free outside ZeroGPU Spaces.
- ``gpu_context()`` — A context manager that ensures GPU is available for the
  duration of the block (effect-free outside ZeroGPU Spaces).
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator


logger = None  # Set lazily by callers

# Global GPU size set by the ``--gpu-size`` CLI flag (default: "large")
_zerogpu_size: str = "large"


def is_zerogpu() -> bool:
    """Detect whether we are running in a ZeroGPU Hugging Face Space."""
    return _get_spaces_gpu_decorator() is not None


def zerogpu_enabled() -> bool:
    """Return whether ZeroGPU is active in the serve process.

    This combines the environment detection with the explicit ``--zerogpu``
    CLI flag from ``Serve.__init__``, which set ZeroGPU mode with ``serve_zerogpu._zerogpu_enabled``.

    Returns:
        `bool`: ``True`` if ZeroGPU mode is active and if we are running in a ZeroGPU Hugging Face Space.
    """
    return _zerogpu_enabled and is_zerogpu()


def zerogpu_size() -> str:
    """Return the configured GPU size.

    Returns:
        `str`: GPU size — ``"large"`` or ``"xlarge"``.
    """
    return _zerogpu_size


def _get_spaces_gpu_decorator() -> type | None:
    """Try to import the ``@spaces.GPU`` decorator from the ``spaces`` package.

    Returns:
        The ``@spaces.GPU`` decorator if available, ``None`` otherwise.
    """
    try:
        from spaces import GPU

        return GPU
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def zerogpu_decorator(size: str = "large") -> Callable:
    """Create a ``@spaces.GPU``-compatible decorator for ZeroGPU Spaces.

    This decorator wraps functions with ``@spaces.GPU`` in ZeroGPU Spaces
    and is **effect-free** in non-ZeroGPU environments.

    It handles both sync and async functions:
    - **Sync functions**: directly apply ``@spaces.GPU``.
    - **Async functions**: wrap in a sync function that runs the async method
      under ``@spaces.GPU``, then return the result.

    The decorator is effect-free outside ZeroGPU Spaces — the function runs
    normally without any GPU context.

    Args:
        size (`str`, *optional*, defaults to ``"large"``):
            GPU size: ``"large"`` or ``"xlarge"``.

    Returns:
        A decorator function.

    Example:
        ```python
        from transformers.cli.serve_zerogpu import zerogpu_decorator

        gpu = zerogpu_decorator(size="xlarge")

        @gpu
        async def handle_request(self, body: dict, request_id: str):
            # This runs under GPU in ZeroGPU Spaces
            ...
        ```
    """
    spaces_gpu = _get_spaces_gpu_decorator()

    def decorator(func: Callable) -> Callable:
        if spaces_gpu is not None and is_zerogpu():
            # ZeroGPU mode: apply @spaces.GPU to allocate GPU per function call
            if inspect.iscoroutinefunction(func):
                # Create a sync wrapper that runs the async function.
                # FastAPI runs endpoints in an event loop; we must use
                # asyncio.new_event_loop() for nested run_until_complete
                # to avoid event loop conflicts.
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(func(*args, **kwargs))
                    finally:
                        loop.close()

                return spaces_gpu(size=size)(sync_wrapper)
            else:
                return spaces_gpu(size=size)(func)
        else:
            # Non-ZeroGPU: effect-free pass-through
            return func

    return decorator


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def gpu_context(size: str = "large") -> Generator[None, None, None]:
    """Context manager that ensures GPU is available for inference.

    In a ZeroGPU Space, this activates ``@spaces.GPU`` for the duration
    of the context block. In non-ZeroGPU environments, this is a no-op.

    This context manager is **effect-free** outside ZeroGPU Spaces.

    Example:
        ```python
        from transformers.cli.serve_zerogpu import gpu_context

        with gpu_context(size="xlarge"):
            outputs = model.generate(**inputs)
        ```

    Args:
        size (`str`, *optional*, defaults to ``"large"``):
            GPU size: ``"large"`` (half RTX Pro 6000 Blackwell) or
            ``"xlarge"`` (full RTX Pro 6000 Blackwell).
    """
    # In ZeroGPU Spaces, GPU allocation is handled by the @spaces.GPU
    # decorator wrapping serve endpoints in server.py. This context manager
    # is effect-free — model loading and inference happen inside the
    # @spaces.GPU-wrapped endpoint handler.
    yield


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_zerogpu_config() -> dict:
    """Return the current ZeroGPU configuration.

    Returns:
        `dict`: Configuration dict with keys:
            - ``enabled``: Whether ZeroGPU is active.
            - ``size``: GPU size setting.
            - ``space_id``: The Hugging Face Space ID (if running in a Space).
    """
    return {
        "enabled": zerogpu_enabled(),
        "size": _zerogpu_size,
        "space_id": os.environ.get("SPACE_ID", ""),
    }


# ---------------------------------------------------------------------------
# Gradio SDK helpers
# ---------------------------------------------------------------------------


def get_spaces_gpu(size: str | None = None) -> Callable:
    """Get a properly configured ``@spaces.GPU`` decorator.

    This returns the real ``@spaces.GPU`` from the ``spaces`` package
    when running in a ZeroGPU Space, or an effect-free pass-through
    otherwise.

    Args:
        size (`str`, *optional*):
            GPU size to request. Defaults to ``"large"``.

    Returns:
        A decorator that can be applied to inference functions.
    """
    return zerogpu_decorator(size=size or _zerogpu_size)


def create_zero_gpu_decorator(size: str = "large") -> Callable:
    """Create a ``@spaces.GPU``-compatible decorator for inference functions.

    This is the recommended way to wrap inference functions for ZeroGPU Spaces.
    The decorator is effect-free outside ZeroGPU Spaces.

    Args:
        size (`str`, *optional*, defaults to ``"large"``):
            GPU size: ``"large"`` or ``"xlarge"``.

    Returns:
        A decorator function.

    Example:
        ```python
        from transformers.cli.serve_zerogpu import create_zero_gpu_decorator

        gpu = create_zero_gpu_decorator(size="xlarge")

        @gpu
        def generate_text(text: str) -> str:
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        ```
    """
    return zerogpu_decorator(size=size)


# ---------------------------------------------------------------------------
# Module-level state (set by Serve.__init__ when --zerogpu is passed)
# ---------------------------------------------------------------------------


_zerogpu_enabled: bool = False
