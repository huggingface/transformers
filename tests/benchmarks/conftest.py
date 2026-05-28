"""
Conftest for benchmarks: provide a no-op ``benchmark`` fixture so that benchmark
tests are skipped (rather than erroring) when ``pytest-benchmark`` is not installed.
"""

import pytest


try:
    import pytest_benchmark  # noqa: F401
except ImportError:
    # Provide a stub fixture that skips gracefully.
    @pytest.fixture
    def benchmark(request):
        pytest.skip("pytest-benchmark not installed (pip install pytest-benchmark)")
