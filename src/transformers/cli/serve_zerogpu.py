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

import inspect
import os
from collections.abc import Callable
from functools import partial


logger = None  # Set lazily by callers

# Global GPU size set by the ``--gpu-size`` CLI flag
_zerogpu_size: str | None = None


def is_zerogpu() -> bool:
    """Detect whether we are running in a ZeroGPU Hugging Face Space."""
    return _get_spaces_gpu_decorator() is not None


def zerogpu_enabled() -> bool:
    """Return whether ZeroGPU is active in the serve process.

    This combines the environment detection (`spaces` and SPACE_ID available) and the explicit ``--zerogpu`` flag.

    Returns:
        `bool`: ``True`` if ZeroGPU mode is active and if we are running in a ZeroGPU Hugging Face Space.
    """
    return _zerogpu_enabled and os.environ.get("SPACE_ID") and is_zerogpu()


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
      under ``@spaces.GPU``, then run the wrapped sync function in a separate
      thread asynchronously.

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
    _in_zerogpu = zerogpu_enabled()

    def decorator(func: Callable) -> Callable:
        if logger is not None:
            if _in_zerogpu:
                logger.warning(
                    "ZeroGPU enabled for '%s' — @spaces.GPU(size='%s') will allocate a GPU per request",
                    func.__name__,
                    size,
                )
            else:
                reason = []
                if spaces_gpu is None:
                    reason.append("'spaces' package not installed")
                if not is_zerogpu():
                    reason.append("not running in a ZeroGPU Space")
                logger.warning(
                    "ZeroGPU NOT applied for '%s' — %s (effect-free pass-through)",
                    func.__name__,
                    "; ".join(reason),
                )
        if _in_zerogpu:
            if inspect.iscoroutinefunction(func):
                import anyio
                import anyio.to_thread

                return partial(anyio.to_thread.run_sync, spaces_gpu(partial(anyio.run, func)))
            else:
                return spaces_gpu(func)
        else:
            # Non-ZeroGPU: effect-free pass-through
            return func

    return decorator


def startup():
    """Since we don't call gradio.launch(), let's start zerogpu manually"""
    try:
        import spaces.zero

        spaces.zero.startup()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Module-level state (set by Serve.__init__ when --zerogpu is passed)
# ---------------------------------------------------------------------------


_zerogpu_enabled: bool = False
