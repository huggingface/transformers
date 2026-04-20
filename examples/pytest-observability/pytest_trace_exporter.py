#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import fsum
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_JAEGER_URL = "http://jaeger:16686"
DEFAULT_LIMIT = 200
DEFAULT_LOOKBACK = "1h"
DEFAULT_PORT = 8000
DEFAULT_RESOURCE_METRICS_FILE = "/data/pytest-resource-metrics.jsonl"
DEFAULT_SERVICE_NAME = "transformers-pytest-demo"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "")
    try:
        return int(value) if value else default
    except ValueError:
        return default


def metric_labels(labels: dict[str, str]) -> str:
    segments = []
    for key, value in labels.items():
        escaped = value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
        segments.append(f'{key}="{escaped}"')
    return "{" + ",".join(segments) + "}"


def tag_map(items: list[dict]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for item in items:
        key = item.get("key")
        value = item.get("value")
        if isinstance(key, str) and value is not None:
            mapped[key] = str(value)
    return mapped


def split_pytest_nodeid(nodeid: str) -> dict[str, str]:
    parts = nodeid.split("::")
    module_path = parts[0] if parts else ""
    module_name = os.path.basename(module_path)
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


def trace_start_time(trace: dict) -> int:
    spans = trace.get("spans", [])
    if not isinstance(spans, list):
        return 0
    return max((int(span.get("startTime", 0)) for span in spans if isinstance(span, dict)), default=0)


def fetch_traces() -> list[dict]:
    base_url = os.getenv("PYTEST_TRACE_EXPORTER_JAEGER_URL", DEFAULT_JAEGER_URL).rstrip("/")
    limit = env_int("PYTEST_TRACE_EXPORTER_LIMIT", DEFAULT_LIMIT)
    lookback = os.getenv("PYTEST_TRACE_EXPORTER_LOOKBACK", DEFAULT_LOOKBACK)
    service_name = os.getenv("PYTEST_TRACE_EXPORTER_SERVICE_NAME", DEFAULT_SERVICE_NAME)
    search_url = (
        f"{base_url}/api/traces?service={quote(service_name)}&limit={limit}&lookback={quote(lookback)}"
    )
    with urlopen(search_url, timeout=5) as response:
        payload = json.load(response)

    traces = payload.get("data", [])
    if not isinstance(traces, list):
        return []
    return [trace for trace in traces if isinstance(trace, dict)]


def fetch_resource_records() -> list[dict[str, str | float | int]]:
    resource_metrics_file = Path(os.getenv("PYTEST_RESOURCE_METRICS_FILE", DEFAULT_RESOURCE_METRICS_FILE))
    if not resource_metrics_file.exists():
        return []

    records: list[dict[str, str | float | int]] = []
    with resource_metrics_file.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def latest_trace(traces: list[dict]) -> dict | None:
    if not traces:
        return None
    return max(traces, key=trace_start_time)


def extract_trace_rows(trace: dict) -> tuple[dict[str, str | int], list[dict[str, str | float]]]:
    trace_id = trace.get("traceID")
    spans = trace.get("spans", [])
    processes = trace.get("processes", {})

    if not isinstance(trace_id, str) or not isinstance(spans, list) or not isinstance(processes, dict):
        return {"trace_id": "unknown", "latest_start_time": 0}, []

    process_job = ""
    process_provider = ""
    service_name = ""
    latest_start_time = 0
    rows: list[dict[str, str | float]] = []

    for span in spans:
        if not isinstance(span, dict):
            continue

        span_start_time = int(span.get("startTime", 0))
        if span_start_time > latest_start_time:
            latest_start_time = span_start_time

        process = processes.get(span.get("processID"), {})
        process_tags = tag_map(process.get("tags", [])) if isinstance(process, dict) else {}
        service_name = process.get("serviceName", service_name) if isinstance(process, dict) else service_name
        process_job = process_tags.get("transformers.test.job", process_job)
        process_provider = process_tags.get("transformers.test.provider", process_provider)

        span_tags = tag_map(span.get("tags", []))
        nodeid = span_tags.get("pytest.nodeid")
        span_type = span_tags.get("pytest.span_type")
        operation_name = span.get("operationName")
        if nodeid is None or span_type != "test" or operation_name != nodeid:
            continue

        node_parts = split_pytest_nodeid(nodeid)
        rows.append(
            {
                "duration_seconds": int(span.get("duration", 0)) / 1_000_000,
                "provider": process_provider or "unknown",
                "service_name": service_name or "unknown",
                "status_code": span_tags.get("otel.status_code", "UNSET"),
                "test_class": node_parts["test_class"],
                "test_function": node_parts["test_function"],
                "test_job": process_job or "unknown",
                "test_module": node_parts["test_module"],
                "test_nodeid": nodeid,
                "trace_id": trace_id,
            }
        )

    return {
        "latest_start_time": latest_start_time,
        "provider": process_provider or "unknown",
        "service_name": service_name or "unknown",
        "test_job": process_job or "unknown",
        "trace_id": trace_id,
    }, rows


def extract_latest_trace_metrics(trace: dict) -> list[str]:
    trace_info, rows = extract_trace_rows(trace)
    lines = [
        "# HELP pytest_test_duration_seconds Duration of pytest test spans from the latest trace.",
        "# TYPE pytest_test_duration_seconds gauge",
        "# HELP pytest_latest_trace_info Metadata for the latest pytest trace visible to the exporter.",
        "# TYPE pytest_latest_trace_info gauge",
    ]

    for row in rows:
        labels = {
            "test_job": str(row["test_job"]),
            "provider": str(row["provider"]),
            "service_name": str(row["service_name"]),
            "status_code": str(row["status_code"]),
            "test_class": str(row["test_class"]),
            "test_function": str(row["test_function"]),
            "test_module": str(row["test_module"]),
            "test_nodeid": str(row["test_nodeid"]),
            "trace_id": str(row["trace_id"]),
        }
        lines.append(f"pytest_test_duration_seconds{metric_labels(labels)} {float(row['duration_seconds']):.9f}")

    info_labels = {
        "test_job": str(trace_info["test_job"]),
        "provider": str(trace_info["provider"]),
        "service_name": str(trace_info["service_name"]),
        "trace_id": str(trace_info["trace_id"]),
    }
    lines.append(f"pytest_latest_trace_info{metric_labels(info_labels)} 1")
    lines.append(
        "pytest_latest_trace_start_time_seconds"
        f"{metric_labels(info_labels)} {int(trace_info['latest_start_time']) / 1_000_000:.6f}"
    )
    return lines


def extract_average_metrics(traces: list[dict]) -> list[str]:
    lines = [
        "# HELP pytest_test_average_duration_seconds Average duration of pytest test spans across fetched traces.",
        "# TYPE pytest_test_average_duration_seconds gauge",
        "# HELP pytest_test_run_count Number of fetched traces that contained a given pytest test span.",
        "# TYPE pytest_test_run_count gauge",
    ]
    aggregates: dict[tuple[str, str, str, str], dict[str, str | list[float]]] = {}

    for trace in traces:
        _, rows = extract_trace_rows(trace)
        for row in rows:
            key = (
                str(row["service_name"]),
                str(row["test_job"]),
                str(row["provider"]),
                str(row["test_nodeid"]),
            )
            if key not in aggregates:
                aggregates[key] = {
                    "durations": [],
                    "test_class": str(row["test_class"]),
                    "test_function": str(row["test_function"]),
                    "test_module": str(row["test_module"]),
                }
            durations = aggregates[key]["durations"]
            assert isinstance(durations, list)
            durations.append(float(row["duration_seconds"]))

    for (service_name, test_job, provider, test_nodeid), aggregate in sorted(aggregates.items()):
        durations = aggregate["durations"]
        assert isinstance(durations, list)
        labels = {
            "test_job": test_job,
            "provider": provider,
            "service_name": service_name,
            "test_class": str(aggregate["test_class"]),
            "test_function": str(aggregate["test_function"]),
            "test_module": str(aggregate["test_module"]),
            "test_nodeid": test_nodeid,
        }
        lines.append(
            f"pytest_test_average_duration_seconds{metric_labels(labels)} {fsum(durations) / len(durations):.9f}"
        )
        lines.append(f"pytest_test_run_count{metric_labels(labels)} {len(durations)}")

    return lines


def extract_average_resource_metrics(records: list[dict[str, str | float | int]]) -> list[str]:
    lines = [
        "# HELP pytest_test_average_cpu_time_seconds Average process CPU time delta across recorded test runs.",
        "# TYPE pytest_test_average_cpu_time_seconds gauge",
        "# HELP pytest_test_average_rss_peak_bytes Average peak RSS across recorded test runs.",
        "# TYPE pytest_test_average_rss_peak_bytes gauge",
        "# HELP pytest_test_average_rss_delta_bytes Average RSS delta across recorded test runs.",
        "# TYPE pytest_test_average_rss_delta_bytes gauge",
        "# HELP pytest_test_average_cuda_peak_allocated_bytes Average peak CUDA allocated bytes across recorded test runs.",
        "# TYPE pytest_test_average_cuda_peak_allocated_bytes gauge",
        "# HELP pytest_test_resource_run_count Number of recorded resource samples for a given test.",
        "# TYPE pytest_test_resource_run_count gauge",
    ]
    aggregates: dict[tuple[str, str, str, str], dict[str, str | list[float]]] = {}

    for record in records:
        service_name = str(record.get("service_name", "unknown"))
        test_job = str(record.get("test_job", "unknown"))
        provider = str(record.get("provider", "unknown"))
        test_nodeid = str(record.get("test_nodeid", "unknown"))
        key = (service_name, test_job, provider, test_nodeid)
        if key not in aggregates:
            aggregates[key] = {
                "cpu_time_seconds": [],
                "rss_delta_bytes": [],
                "rss_peak_bytes": [],
                "cuda_peak_allocated_bytes": [],
                "test_class": str(record.get("test_class", "")),
                "test_function": str(record.get("test_function", "")),
                "test_module": str(record.get("test_module", "")),
            }

        aggregate = aggregates[key]
        for metric_name in ("cpu_time_seconds", "rss_delta_bytes", "rss_peak_bytes", "cuda_peak_allocated_bytes"):
            value = record.get(metric_name)
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if isinstance(value, (int, float)):
                metric_values.append(float(value))

    for (service_name, test_job, provider, test_nodeid), aggregate in sorted(aggregates.items()):
        labels = {
            "test_job": test_job,
            "provider": provider,
            "service_name": service_name,
            "test_class": str(aggregate["test_class"]),
            "test_function": str(aggregate["test_function"]),
            "test_module": str(aggregate["test_module"]),
            "test_nodeid": test_nodeid,
        }
        resource_count = len(aggregate["cpu_time_seconds"])  # type: ignore[arg-type]
        lines.append(f"pytest_test_resource_run_count{metric_labels(labels)} {resource_count}")
        for metric_name, prom_name in (
            ("cpu_time_seconds", "pytest_test_average_cpu_time_seconds"),
            ("rss_peak_bytes", "pytest_test_average_rss_peak_bytes"),
            ("rss_delta_bytes", "pytest_test_average_rss_delta_bytes"),
            ("cuda_peak_allocated_bytes", "pytest_test_average_cuda_peak_allocated_bytes"),
        ):
            metric_values = aggregate[metric_name]
            assert isinstance(metric_values, list)
            if not metric_values:
                continue
            lines.append(f"{prom_name}{metric_labels(labels)} {fsum(metric_values) / len(metric_values):.9f}")

    return lines


def render_metrics() -> str:
    try:
        traces = fetch_traces()
        resource_records = fetch_resource_records()
    except Exception as error:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 0\n"
            "# HELP pytest_trace_exporter_last_error Last exporter error.\n"
            "# TYPE pytest_trace_exporter_last_error gauge\n"
            f"pytest_trace_exporter_last_error{{message={json.dumps(str(error))}}} 1\n"
        )

    if not traces:
        return (
            "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.\n"
            "# TYPE pytest_trace_exporter_up gauge\n"
            "pytest_trace_exporter_up 1\n"
        )

    rendered = [
        "# HELP pytest_trace_exporter_up Whether the exporter could query Jaeger.",
        "# TYPE pytest_trace_exporter_up gauge",
        "pytest_trace_exporter_up 1",
        "# HELP pytest_trace_exporter_trace_count Number of traces fetched from Jaeger for aggregation.",
        "# TYPE pytest_trace_exporter_trace_count gauge",
        f"pytest_trace_exporter_trace_count {len(traces)}",
    ]
    rendered.extend(extract_average_metrics(traces))
    rendered.extend(extract_average_resource_metrics(resource_records))

    latest = latest_trace(traces)
    if latest is not None:
        rendered.extend(extract_latest_trace_metrics(latest))
    return "\n".join(rendered) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/metrics", "/"}:
            self.send_response(404)
            self.end_headers()
            return

        payload = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def main() -> None:
    port = env_int("PYTEST_TRACE_EXPORTER_PORT", DEFAULT_PORT)
    server = ThreadingHTTPServer(("0.0.0.0", port), MetricsHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
