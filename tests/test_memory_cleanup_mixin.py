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

import math
import os
import warnings
from collections.abc import Iterable
from typing import Any

from transformers.testing_utils import (
    TestCasePlus,
    backend_max_memory_allocated,
    backend_memory_allocated,
    backend_reset_peak_memory_stats,
    cleanup,
    torch_device,
)
from transformers.utils import is_psutil_available, is_torch_available


if is_torch_available():
    import torch


# Attributes `unittest` may add after `setUp` ran, so not the test's own.
_UNITTEST_INTERNAL_ATTRS = frozenset({"_outcome", "_subtest", "_cleanups", "_type_equality_funcs"})

# `MemoryCleanupMixin`'s own bookkeeping.
_MEMORY_CLEANUP_ATTRS = frozenset(
    {
        "_memory_cleanup_class_attrs",
        "_memory_cleanup_instance_attrs",
        "_memory_cleanup_baseline",
        "_memory_cleanup_rss_baseline",
    }
)


def with_grad(method):
    """Run this test method with autograd on, whatever the class default is."""
    method._memory_cleanup_no_grad = False
    return method


def with_no_grad(method):
    """Run this test method under `torch.no_grad()`, whatever the class default is."""
    method._memory_cleanup_no_grad = True
    return method


def _memory_leak_settings() -> tuple[float | None, str]:
    """Return `(threshold_mib, mode)` from the environment; `threshold_mib` is `None` when the check is off."""
    raw = os.environ.get("TRANSFORMERS_TEST_MEMORY_LEAK_MIB", "").strip()
    if not raw:
        return None, "warn"
    try:
        threshold = float(raw)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"`TRANSFORMERS_TEST_MEMORY_LEAK_MIB` must be a number of MiB, got {raw!r}.") from e
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"`TRANSFORMERS_TEST_MEMORY_LEAK_MIB` must be finite and non-negative, got {raw!r}.")
    mode = os.environ.get("TRANSFORMERS_TEST_MEMORY_LEAK_MODE", "warn").strip().lower()
    if mode not in ("warn", "error"):
        raise ValueError(f"`TRANSFORMERS_TEST_MEMORY_LEAK_MODE` must be 'warn' or 'error', got {mode!r}.")
    return threshold, mode


class MemoryCleanupMixin:
    """
    Frees the memory a test class allocates, so one test's leftovers cannot OOM the next.

    - Runs `cleanup(torch_device, gc_collect=True)` before and after every test.
    - Deletes attributes the test added to `self` and to the class, `@cached_property` caches included: pytest keeps
      test instances alive for the whole session, so `gc.collect()` cannot free what they still reference.
    - Runs test methods under `torch.no_grad()`, since a forward pass otherwise retains activations. Set
      `cleanup_no_grad = False` on a class that trains, or use [`with_grad`] / [`with_no_grad`] per method.

    Put it first in the bases. `MemoryCleanupTestCase` pairs it with `TestCasePlus`.

    ```python
    class MyModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
        def test_generation(self):
            self.model = AutoModelForCausalLM.from_pretrained(...).to(torch_device)  # dropped in tearDown
    ```

    Attributes assigned in the class body are kept; everything added later is dropped. An overridden `setUp` must
    call `super().setUp()` (the instance snapshot is taken there) or the test errors out saying so.

    Leak check, off by default since collecting frees what a reproducer needs: `TRANSFORMERS_TEST_MEMORY_LEAK_MIB=<n>`
    reports tests leaving more than `<n>` MiB on the device, `TRANSFORMERS_TEST_MEMORY_LEAK_MODE=error` fails them.

    Known leak, still unfixed: compiling with `cache_implementation="static"` leaves memory in the cache.
    """

    cleanup_no_grad: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Snapshotting at class-creation time keeps the class body (and only it), so `setUpClass` needs no
        # particular ordering -- whatever it assigns is later dropped.
        snapshot = set(vars(cls))
        cls._memory_cleanup_class_attrs = snapshot

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            # `setUpClass` may have parked a model on the class; nothing else drops it.
            _drop_new_attributes(cls, cls._memory_cleanup_class_attrs)
            _run_cleanup()

    def setUp(self):
        super().setUp()
        _run_cleanup()  # a previous class may have left memory behind
        self._memory_cleanup_instance_attrs = set(vars(self))
        if _memory_leak_settings()[0] is not None:
            self._memory_cleanup_baseline = _device_memory_allocated()
            self._memory_cleanup_rss_baseline = _process_rss()
            if is_torch_available():
                backend_reset_peak_memory_stats(torch_device)

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            known = getattr(self, "_memory_cleanup_instance_attrs", None)
            if known is not None:
                _drop_new_attributes(self, known)
            _run_cleanup()
            self._check_for_memory_leak()

    def _callTestMethod(self, method):
        if not hasattr(self, "_memory_cleanup_instance_attrs"):
            raise RuntimeError(
                f"{type(self).__name__}.setUp() must call super().setUp(): MemoryCleanupMixin snapshots the "
                "instance attributes there, and cannot release the test's references without it."
            )
        # Private hook, but the only seam wrapping the test method without `setUp`, where a loaded model must keep
        # its `requires_grad`. `MemoryCleanupUnderPytestTest` fails if a runner stops routing through it.
        if getattr(method, "_memory_cleanup_no_grad", self.cleanup_no_grad) and is_torch_available():
            with torch.no_grad():
                return super()._callTestMethod(method)
        return super()._callTestMethod(method)

    def _check_for_memory_leak(self):
        threshold_mib, mode = _memory_leak_settings()
        if threshold_mib is None:
            return
        leaked_mib = (_device_memory_allocated() - getattr(self, "_memory_cleanup_baseline", 0)) / 1024**2
        if leaked_mib <= threshold_mib:
            return
        rss_delta_mib = (_process_rss() - getattr(self, "_memory_cleanup_rss_baseline", 0)) / 1024**2
        peak_mib = backend_max_memory_allocated(torch_device) / 1024**2 if is_torch_available() else 0
        message = (
            f"{self.id()} left {leaked_mib:.1f} MiB allocated on {torch_device} after teardown "
            f"(threshold {threshold_mib:.1f} MiB, peak during the test {peak_mib:.1f} MiB, "
            f"CPU RSS {rss_delta_mib:+.1f} MiB). "
            "Something still references a device tensor: a model on `self`/the class, captured by a closure, or "
            "held by a `@cached_property`."
        )
        if mode == "error":
            raise AssertionError(message)
        warnings.warn(message, stacklevel=2)


class MemoryCleanupTestCase(MemoryCleanupMixin, TestCasePlus):
    """`TestCasePlus` plus `MemoryCleanupMixin`, for integration tests that load real checkpoints."""


def _run_cleanup() -> None:
    """`cleanup(torch_device, gc_collect=True)`, but a no-op instead of a skip when torch is missing."""
    if is_torch_available():
        cleanup(torch_device, gc_collect=True)


def _device_memory_allocated() -> int:
    """Bytes currently allocated on `torch_device`; `0` on backends that do not report it (including CPU)."""
    if not is_torch_available():
        return 0
    return backend_memory_allocated(torch_device) or 0


def _process_rss() -> int:
    """Resident set size of this process in bytes; `0` when `psutil` is not installed."""
    if not is_psutil_available():
        return 0
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss


def _drop_new_attributes(obj: Any, known: Iterable[str]) -> None:
    """Delete the attributes `obj` gained since `known` was snapshotted, so their referents can be collected."""
    protected = set(known) | _UNITTEST_INTERNAL_ATTRS | _MEMORY_CLEANUP_ATTRS
    for name in list(vars(obj)):
        if name in protected:
            continue
        try:
            delattr(obj, name)
        except AttributeError:
            # Read-only or already gone (e.g. a slot, or a descriptor on a parent class).
            pass
