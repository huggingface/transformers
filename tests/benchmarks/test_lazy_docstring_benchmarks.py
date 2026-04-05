# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Benchmarks for the lazy-docstring machinery introduced in ``auto_docstring.py``.

Run with::

    pip install pytest-benchmark
    pytest tests/benchmarks/test_lazy_docstring_benchmarks.py -v --benchmark-only

These benchmarks are **informational** — they assert nothing about absolute
thresholds.  Use them to compare before/after performance of ``auto_docstring``
changes, or to spot regressions in import / doc-access paths.
"""

import importlib
import sys

import pytest


try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

pytestmark = pytest.mark.skipif(
    not HAS_BENCHMARK, reason="pytest-benchmark not installed (pip install pytest-benchmark)"
)


# ---------------------------------------------------------------------------
# 1. Module import time
# ---------------------------------------------------------------------------


def _do_import_image_processing():
    """Re-import ``image_processing_utils`` from scratch each round."""
    sys.modules.pop("transformers.image_processing_utils", None)
    importlib.import_module("transformers.image_processing_utils")


@pytest.mark.benchmark(group="import")
def test_import_image_processing(benchmark):
    """Measure how long it takes to import ``transformers.image_processing_utils``.

    A significant portion of this time used to be docstring generation; with the
    lazy approach that cost is deferred until ``__doc__`` is first accessed.
    """
    # Warm-up: ensure everything except the target module is already cached.
    import transformers.image_processing_utils  # noqa: F401

    benchmark(_do_import_image_processing)


# ---------------------------------------------------------------------------
# 2. Class ``__doc__`` access — first (generates) vs cached
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="doc_access")
def test_class_doc_first_access(benchmark):
    """Measure the cost of the *first* ``cls.__doc__`` access (triggers generation).

    Because ``_LazyDocClass.__get__`` replaces itself with a plain string after the
    first call, subsequent benchmarks in this process will measure the cached path.
    Run with ``--benchmark-disable-gc`` for reproducible timings.
    """
    from transformers.image_processing_utils import BaseImageProcessor

    # Reset the lazy state so every round re-generates.
    from transformers.utils.auto_docstring import auto_class_docstring

    def setup():
        auto_class_docstring(BaseImageProcessor)

    def access():
        return BaseImageProcessor.__doc__

    benchmark.pedantic(access, setup=setup, rounds=10, iterations=1)


@pytest.mark.benchmark(group="doc_access")
def test_class_doc_cached_access(benchmark):
    """Measure the cost of accessing ``cls.__doc__`` after it has been generated.

    After the first access the lazy descriptor replaces itself with a plain string,
    so this path should be essentially free.
    """
    from transformers.image_processing_utils import BaseImageProcessor

    # Ensure doc is already generated (cached).
    _ = BaseImageProcessor.__doc__

    benchmark(lambda: BaseImageProcessor.__doc__)


# ---------------------------------------------------------------------------
# 3. Method ``__doc__`` access
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="doc_access")
def test_method_doc_access(benchmark):
    """Measure ``method.__doc__`` access cost after eager decoration.

    Methods are decorated eagerly (``func.__doc__`` is set at decoration time and
    the original function is returned unchanged).  Subsequent reads are a plain
    attribute lookup — essentially free.
    """
    from transformers.utils.auto_docstring import auto_method_docstring

    def _dummy(x: int, y: int = 0) -> int:
        r"""x (`int`): First number.\ny (`int`, *optional*): Second number."""
        return x + y

    _dummy.__qualname__ = "DummyClass.forward"  # appear as a method to auto_method_docstring
    auto_method_docstring(_dummy)

    benchmark(lambda: _dummy.__doc__)


# ---------------------------------------------------------------------------
# 4. ``from_pretrained`` with a tiny model (end-to-end smoke benchmark)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="from_pretrained")
@pytest.mark.slow
def test_from_pretrained_tiny_llama(benchmark):
    """Measure ``LlamaForCausalLM.from_pretrained`` on a tiny random model.

    This is a *slow* benchmark (marked with ``@pytest.mark.slow``) that requires
    network access and PyTorch.  It is skipped by default unless ``RUN_SLOW=1``
    is set.  Run with::

        RUN_SLOW=1 pytest tests/benchmarks/test_lazy_docstring_benchmarks.py \
            -k test_from_pretrained_tiny_llama -v --benchmark-only
    """
    import os

    if not os.environ.get("RUN_SLOW"):
        pytest.skip("Set RUN_SLOW=1 to run this benchmark")

    try:
        from transformers import LlamaForCausalLM
    except ImportError:
        pytest.skip("PyTorch is required for this benchmark")

    benchmark(
        LlamaForCausalLM.from_pretrained,
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        low_cpu_mem_usage=False,
    )
