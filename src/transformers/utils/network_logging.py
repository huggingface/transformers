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

from __future__ import annotations

import inspect
import json
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

import httpx

from .generic import strtobool


class _NetworkRequestTrace:
    def __init__(self, request: httpx.Request):
        self.request = request
        self.started_at = time.perf_counter()
        self.phase_started_at = {}
        self.phases_ms = defaultdict(float)

    def trace(self, name: str, info: dict[str, Any]) -> None:
        parts = name.rsplit(".", 2)
        if len(parts) != 3:
            return

        _, phase, state = parts
        now = time.perf_counter()
        if state == "started":
            self.phase_started_at[phase] = now
        elif state in {"complete", "failed"}:
            phase_started_at = self.phase_started_at.pop(phase, None)
            if phase_started_at is not None:
                self.phases_ms[phase] += (now - phase_started_at) * 1000

    def build_record(
        self,
        *,
        response: httpx.Response | None = None,
        error: BaseException | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        total_ms = (time.perf_counter() - self.started_at) * 1000
        url = self.request.url
        host = url.host or ""
        port = url.port
        default_port = {"http": 80, "https": 443}.get(url.scheme)
        host_display = host if port in (None, default_port) else f"{host}:{port}"

        http_version = None
        status_code = None
        bytes_downloaded = None
        response_complete = False
        if response is not None:
            status_code = response.status_code
            response_complete = response.is_closed
            raw_http_version = response.extensions.get("http_version")
            if isinstance(raw_http_version, bytes):
                http_version = raw_http_version.decode("ascii", errors="replace")
            elif raw_http_version is not None:
                http_version = str(raw_http_version)

            if response_complete:
                try:
                    bytes_downloaded = len(response.content)
                except httpx.ResponseNotRead:
                    pass

        return {
            "method": self.request.method,
            "scheme": url.scheme,
            "host": host,
            "host_display": host_display,
            "port": port,
            "path": url.path,
            "has_query": bool(url.query),
            "url": f"{url.scheme}://{host_display}{url.path}{'?...' if url.query else ''}",
            "request_id": self.request.headers.get("x-amzn-trace-id") or self.request.headers.get("x-request-id"),
            "status_code": status_code,
            "http_version": http_version,
            "bytes_downloaded": bytes_downloaded,
            "total_ms": total_ms,
            "stream": stream,
            "response_complete": response_complete,
            "phases_ms": dict(sorted(self.phases_ms.items())),
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }


class _NetworkDebugProfiler:
    def __init__(self):
        self._records = []
        self._lock = threading.Lock()
        self._enabled = False
        self._output_path = None
        self._original_client_send = None
        self._original_async_client_send = None
        self._shared_dir = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def clear(self) -> None:
        with self._lock:
            self._records = []

    def enable(self, output_path: str | os.PathLike | None = None) -> None:
        if self._enabled:
            self._output_path = None if output_path is None else os.fspath(output_path)
            self.clear()
            return

        self._output_path = None if output_path is None else os.fspath(output_path)
        self.clear()

        profiler = self
        self._original_client_send = httpx.Client.send
        self._original_async_client_send = httpx.AsyncClient.send

        @wraps(self._original_client_send)
        def patched_client_send(client, request, *args, **kwargs):
            return profiler._send_with_trace(profiler._original_client_send, client, request, *args, **kwargs)

        @wraps(self._original_async_client_send)
        async def patched_async_client_send(client, request, *args, **kwargs):
            return await profiler._async_send_with_trace(
                profiler._original_async_client_send, client, request, *args, **kwargs
            )

        httpx.Client.send = patched_client_send
        httpx.AsyncClient.send = patched_async_client_send
        self._enabled = True

    def setup_shared_dir(self) -> str | None:
        """Create a shared temp directory for xdist workers to dump records into."""
        if self._shared_dir is None:
            import tempfile

            self._shared_dir = tempfile.mkdtemp(prefix="network_debug_")
        return self._shared_dir

    def set_shared_dir(self, shared_dir: str) -> None:
        """Set the shared directory (called in xdist workers)."""
        self._shared_dir = shared_dir

    def dump_worker_records(self, worker_id: str | None = None) -> None:
        """Write this process's records to a file in the shared directory (called in workers)."""
        if not self._shared_dir or not self._records:
            return
        worker_id = worker_id or f"pid{os.getpid()}"
        dump_path = os.path.join(self._shared_dir, f"records_{worker_id}.json")
        with self._lock:
            records = [{**record, "phases_ms": dict(record["phases_ms"])} for record in self._records]
        Path(dump_path).write_text(json.dumps(records), encoding="utf-8")

    def load_worker_records(self) -> None:
        """Load all worker record files from the shared directory (called in controller)."""
        if not self._shared_dir or not os.path.isdir(self._shared_dir):
            return
        import glob as glob_module

        for record_file in glob_module.glob(os.path.join(self._shared_dir, "records_*.json")):
            try:
                records = json.loads(Path(record_file).read_text(encoding="utf-8"))
                with self._lock:
                    for record in records:
                        record["phases_ms"] = defaultdict(float, record.get("phases_ms", {}))
                        self._records.append(record)
            except (OSError, json.JSONDecodeError):
                pass

    def cleanup_shared_dir(self) -> None:
        """Remove the shared temp directory."""
        if self._shared_dir and os.path.isdir(self._shared_dir):
            import shutil

            shutil.rmtree(self._shared_dir, ignore_errors=True)
            self._shared_dir = None

    def disable(self) -> None:
        if not self._enabled:
            return

        httpx.Client.send = self._original_client_send
        httpx.AsyncClient.send = self._original_async_client_send
        self._enabled = False
        self._original_client_send = None
        self._original_async_client_send = None
        self._output_path = None
        self.clear()

    def _append_record(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)

    def _wrap_trace_callback(self, request: httpx.Request, trace: _NetworkRequestTrace):
        existing_trace = request.extensions.get("trace")

        def wrapped_trace(name: str, info: dict[str, Any]) -> Any:
            trace.trace(name, info)
            if existing_trace is not None:
                return existing_trace(name, info)
            return None

        return wrapped_trace

    async def _awrap_trace_callback(self, request: httpx.Request, trace: _NetworkRequestTrace):
        existing_trace = request.extensions.get("trace")

        async def wrapped_trace(name: str, info: dict[str, Any]) -> Any:
            trace.trace(name, info)
            if existing_trace is not None:
                result = existing_trace(name, info)
                if inspect.isawaitable(result):
                    return await result
                return result
            return None

        return wrapped_trace

    def _send_with_trace(self, original_send, client, request: httpx.Request, *args, **kwargs):
        trace = _NetworkRequestTrace(request)
        request.extensions = dict(request.extensions)
        request.extensions["trace"] = self._wrap_trace_callback(request, trace)

        try:
            response = original_send(client, request, *args, **kwargs)
        except Exception as error:
            self._append_record(trace.build_record(error=error, stream=kwargs.get("stream", False)))
            raise

        self._append_record(trace.build_record(response=response, stream=kwargs.get("stream", False)))
        return response

    async def _async_send_with_trace(self, original_send, client, request: httpx.Request, *args, **kwargs):
        trace = _NetworkRequestTrace(request)
        request.extensions = dict(request.extensions)
        request.extensions["trace"] = await self._awrap_trace_callback(request, trace)

        try:
            response = await original_send(client, request, *args, **kwargs)
        except Exception as error:
            self._append_record(trace.build_record(error=error, stream=kwargs.get("stream", False)))
            raise

        self._append_record(trace.build_record(response=response, stream=kwargs.get("stream", False)))
        return response

    def build_report(self) -> dict[str, Any]:
        with self._lock:
            records = [
                {
                    **record,
                    "phases_ms": dict(record["phases_ms"]),
                }
                for record in self._records
            ]

        phase_totals_ms = defaultdict(float)
        route_totals = {}
        for record in records:
            for phase, duration_ms in record["phases_ms"].items():
                phase_totals_ms[phase] += duration_ms

            route_key = (record["method"], record["host_display"], record["path"])
            route_total = route_totals.setdefault(
                route_key,
                {
                    "method": record["method"],
                    "host_display": record["host_display"],
                    "path": record["path"],
                    "count": 0,
                    "failures": 0,
                    "total_ms": 0.0,
                    "phase_totals_ms": defaultdict(float),
                },
            )
            route_total["count"] += 1
            route_total["total_ms"] += record["total_ms"]
            route_total["failures"] += int(record["error"] is not None)
            for phase, duration_ms in record["phases_ms"].items():
                route_total["phase_totals_ms"][phase] += duration_ms

        routes = []
        for route_total in route_totals.values():
            route_total["avg_ms"] = route_total["total_ms"] / route_total["count"]
            route_total["phase_totals_ms"] = dict(sorted(route_total["phase_totals_ms"].items()))
            routes.append(route_total)

        routes.sort(key=lambda route: route["total_ms"], reverse=True)
        total_time_ms = sum(record["total_ms"] for record in records)
        return {
            "enabled": self._enabled,
            "output_path": self._output_path,
            "total_requests": len(records),
            "failed_requests": sum(int(record["error"] is not None) for record in records),
            "total_time_ms": total_time_ms,
            "phase_totals_ms": dict(sorted(phase_totals_ms.items())),
            "requests": records,
            "routes": routes,
        }

    def maybe_write_report(self) -> str | None:
        if self._output_path is None:
            return None

        report_path = Path(self._output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.build_report(), indent=2, sort_keys=True), encoding="utf-8")
        return str(report_path)


_NETWORK_DEBUG_PROFILER = _NetworkDebugProfiler()


_DEFAULT_REPORT_PATH = "network_debug_report.json"


def _parse_network_debug_env() -> tuple[bool, str]:
    enabled_raw = os.environ.get("NETWORK_DEBUG_REPORT", "").strip()
    try:
        enabled = bool(strtobool(enabled_raw)) if enabled_raw else False
    except ValueError:
        enabled = False

    output_path = os.environ.get("NETWORK_DEBUG_REPORT_PATH", "").strip() or _DEFAULT_REPORT_PATH
    return enabled, output_path


def _enable_network_debug_report(output_path: str | os.PathLike | None = None) -> None:
    _NETWORK_DEBUG_PROFILER.enable(output_path=output_path)


def _disable_network_debug_report() -> None:
    _NETWORK_DEBUG_PROFILER.disable()


def _clear_network_debug_report() -> None:
    _NETWORK_DEBUG_PROFILER.clear()


def _get_network_debug_report() -> dict[str, Any]:
    return _NETWORK_DEBUG_PROFILER.build_report()


def _enable_network_debug_report_from_env() -> bool:
    enabled, output_path = _parse_network_debug_env()
    if not enabled:
        return False

    _enable_network_debug_report(output_path=output_path)
    return True


def _format_network_debug_report(max_requests: int = 20, max_routes: int = 10) -> str:
    report = _get_network_debug_report()
    if report["total_requests"] == 0:
        return "Network debug report: no httpx requests captured."

    lines = [
        "Network debug report",
        f"Requests captured: {report['total_requests']}",
        f"Failed requests: {report['failed_requests']}",
        f"Cumulative request time: {report['total_time_ms']:.1f} ms",
    ]

    if report["phase_totals_ms"]:
        phase_summary = ", ".join(
            f"{phase}={duration_ms:.1f} ms"
            for phase, duration_ms in sorted(report["phase_totals_ms"].items(), key=lambda item: item[1], reverse=True)
        )
        lines.append(f"Phase totals: {phase_summary}")

    lines.append("")
    lines.append("Slowest requests:")
    for idx, record in enumerate(
        sorted(report["requests"], key=lambda request: request["total_ms"], reverse=True)[:max_requests],
        start=1,
    ):
        status = record["error"] or f"status={record['status_code']}"
        phase_bits = []
        for phase in ("connect_tcp", "start_tls", "receive_response_headers", "receive_response_body"):
            duration_ms = record["phases_ms"].get(phase)
            if duration_ms is not None:
                phase_bits.append(f"{phase}={duration_ms:.1f} ms")
        phase_suffix = f" ({', '.join(phase_bits)})" if phase_bits else ""
        incomplete_suffix = " incomplete" if record["stream"] and not record["response_complete"] else ""
        lines.append(
            f"{idx:>2}. {record['method']} {record['url']} {record['total_ms']:.1f} ms {status}{incomplete_suffix}{phase_suffix}"
        )

    lines.append("")
    lines.append("Slowest routes:")
    for idx, route in enumerate(report["routes"][:max_routes], start=1):
        lines.append(
            f"{idx:>2}. {route['method']} {route['host_display']}{route['path']} count={route['count']} "
            f"total={route['total_ms']:.1f} ms avg={route['avg_ms']:.1f} ms failures={route['failures']}"
        )

    return "\n".join(lines)


class NetworkDebugPlugin:
    """Pytest plugin that handles all network debug orchestration including xdist coordination."""

    def pytest_configure(self, config):
        _enable_network_debug_report_from_env()
        if not _NETWORK_DEBUG_PROFILER.enabled:
            return

        # xdist controller: create shared dir for workers to dump network records
        if not hasattr(config, "workerinput"):
            shared_dir = _NETWORK_DEBUG_PROFILER.setup_shared_dir()
            if shared_dir:
                config._network_debug_shared_dir = shared_dir
        else:
            # xdist worker: receive shared dir from controller
            shared_dir = config.workerinput.get("network_debug_shared_dir")
            if shared_dir:
                _NETWORK_DEBUG_PROFILER.set_shared_dir(shared_dir)

    def pytest_configure_node(self, node):
        """xdist hook: called on the controller to configure each worker node."""
        shared_dir = getattr(node.config, "_network_debug_shared_dir", None)
        if shared_dir:
            node.workerinput["network_debug_shared_dir"] = shared_dir

    def pytest_sessionfinish(self, session, exitstatus):
        # xdist worker: dump network debug records for the controller to aggregate
        if hasattr(session.config, "workerinput"):
            worker_id = session.config.workerinput.get("workerid", f"pid{os.getpid()}")
            _NETWORK_DEBUG_PROFILER.dump_worker_records(worker_id=worker_id)

    def pytest_terminal_summary(self, terminalreporter):
        if not _NETWORK_DEBUG_PROFILER.enabled:
            return

        # Skip report generation in xdist worker processes; only the controller should aggregate and report.
        if hasattr(terminalreporter.config, "workerinput"):
            return

        # Aggregate worker records if running under xdist.
        _NETWORK_DEBUG_PROFILER.load_worker_records()

        report_path = None
        try:
            report_path = _NETWORK_DEBUG_PROFILER.maybe_write_report()
        except OSError as error:
            report_path = f"Failed to write JSON report: {error}"

        terminalreporter.section("Network debug", sep="=")
        for line in _format_network_debug_report().splitlines():
            terminalreporter.write_line(line)
        if report_path is not None:
            terminalreporter.write_line(f"JSON report: {report_path}")

        _NETWORK_DEBUG_PROFILER.cleanup_shared_dir()


def register_network_debug_plugin(config) -> None:
    """Register the network debug pytest plugin. Single entry point for conftest.py."""
    config.pluginmanager.register(NetworkDebugPlugin(), "network_debug")


__all__ = [
    "register_network_debug_plugin",
]
