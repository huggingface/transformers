from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest


try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def detect_provider() -> str:
    if os.getenv("GITHUB_ACTIONS"):
        return "github_actions"
    if os.getenv("CIRCLECI") or os.getenv("CIRCLE_WORKFLOW_ID"):
        return "circleci"
    return "local"


def split_pytest_nodeid(nodeid: str) -> dict[str, str]:
    parts = nodeid.split("::")
    module_name = Path(parts[0]).name if parts else ""
    if len(parts) >= 3:
        class_name = parts[-2]
        function_name = parts[-1]
    elif len(parts) == 2:
        class_name = ""
        function_name = parts[-1]
    else:
        class_name = ""
        function_name = ""

    return {
        "test_class": class_name,
        "test_function": function_name,
        "test_module": module_name,
    }


class ResourceSampler:
    def __init__(self) -> None:
        if psutil is None:  # pragma: no cover
            raise RuntimeError("psutil is required for pytest resource monitoring")

        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.start_cpu_time = self._cpu_time_seconds()
        self.start_rss_bytes = self._rss_bytes()
        self.peak_rss_bytes = self.start_rss_bytes
        self.cuda_available = bool(torch is not None and torch.cuda.is_available())
        self.start_cuda_allocated_bytes = self._cuda_allocated_bytes()
        self.peak_cuda_allocated_bytes = self.start_cuda_allocated_bytes

    def _cpu_time_seconds(self) -> float:
        cpu_times = self.process.cpu_times()
        return float(cpu_times.user + cpu_times.system)

    def _rss_bytes(self) -> int:
        return int(self.process.memory_info().rss)

    def _cuda_allocated_bytes(self) -> int:
        if not self.cuda_available:
            return 0
        try:
            return int(torch.cuda.memory_allocated())  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            return 0

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(0.05):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self._rss_bytes())
            if self.cuda_available:
                self.peak_cuda_allocated_bytes = max(self.peak_cuda_allocated_bytes, self._cuda_allocated_bytes())

    def start(self) -> None:
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, float | int]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

        end_rss_bytes = self._rss_bytes()
        end_cpu_time = self._cpu_time_seconds()
        end_cuda_allocated_bytes = self._cuda_allocated_bytes()
        self.peak_rss_bytes = max(self.peak_rss_bytes, end_rss_bytes)
        self.peak_cuda_allocated_bytes = max(self.peak_cuda_allocated_bytes, end_cuda_allocated_bytes)

        return {
            "cpu_time_seconds": max(0.0, end_cpu_time - self.start_cpu_time),
            "rss_delta_bytes": end_rss_bytes - self.start_rss_bytes,
            "rss_end_bytes": end_rss_bytes,
            "rss_peak_bytes": self.peak_rss_bytes,
            "cuda_end_allocated_bytes": end_cuda_allocated_bytes,
            "cuda_peak_allocated_bytes": self.peak_cuda_allocated_bytes,
        }


def metrics_file_path() -> Path | None:
    raw_path = os.getenv("TRANSFORMERS_TEST_RESOURCE_METRICS_FILE")
    if not raw_path:
        return None
    return Path(raw_path)


def write_resource_record(item: pytest.Item, metrics: dict[str, float | int]) -> None:
    path = metrics_file_path()
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    node_parts = split_pytest_nodeid(item.nodeid)
    record: dict[str, Any] = {
        "provider": detect_provider(),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "transformers-tests"),
        "test_job": os.getenv("TRANSFORMERS_TEST_OTEL_JOB_NAME", "local_pytest"),
        "test_nodeid": item.nodeid,
        "timestamp": time.time(),
        **node_parts,
        **metrics,
    }
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    if psutil is None:
        yield
        return

    sampler = ResourceSampler()
    sampler.start()
    outcome = yield
    metrics = sampler.stop()
    write_resource_record(item, metrics)
    return outcome
