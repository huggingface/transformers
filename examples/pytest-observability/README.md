# Pytest Observability

This directory is a self-contained local stack for pytest tracing and slow-test analysis.

It includes:

- `docker-compose.yml`: starts Grafana, Jaeger, Prometheus, the OTEL collector, and a small pytest trace exporter
- `otelcol.yaml`: OTLP receiver configuration for the local collector
- `prometheus.yml`: scrape config for the pytest trace exporter
- `grafana-datasources.yaml`: provisioned Jaeger and Prometheus data sources
- `grafana-dashboard.yaml`: dashboard provisioning
- `pytest-observability-dashboard.json`: the Grafana dashboard
- `pytest_resource_monitor.py`: optional pytest plugin that records per-test CPU and memory samples
- `pytest_trace_exporter.py`: converts Jaeger traces into Prometheus metrics for slow-test ranking
- `data/`: shared local data directory for resource metrics JSONL output

The stack works like this:

- `pytest-opentelemetry` sends spans to the local OTEL collector
- the collector forwards traces to Jaeger
- the optional pytest resource plugin writes per-test CPU and memory samples to `data/pytest-resource-metrics.jsonl`
- the exporter reads recent Jaeger traces plus resource samples and emits Prometheus metrics
- Grafana uses:
  - Jaeger for trace lookup
  - Prometheus for the slow-test table and resource charts

## Prerequisites

- Docker with `docker compose`
- A local environment with `pytest-opentelemetry` available

From the repo root:

```sh
pip install -e ".[testing,test-observability]"
```

## Start the stack

```sh
cd examples/pytest-observability
docker compose up -d
```

## Stop the stack

```sh
cd examples/pytest-observability
docker compose down
```

## Services

- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9091`
- Jaeger: `http://localhost:16687`
- OTLP gRPC collector endpoint: `http://localhost:5317`
- OTLP HTTP collector endpoint: `http://localhost:5318`

## Run one traced pytest job

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=transformers-pytest-demo \
.venv/bin/python utils/configure_ci_otel.py \
  --job-name grafana_demo \
  -- \
  .venv/bin/python -m pytest tests/utils/test_configure_ci_otel.py -q
```

The wrapper prints the root trace ID at the beginning and end of the run:

```text
OTEL TRACE START trace_id=<trace_id> ...
OTEL TRACE END trace_id=<trace_id> ...
```

## Run one traced pytest job with CPU and memory sampling

```sh
PYTHONPATH=examples/pytest-observability \
TRANSFORMERS_TEST_RESOURCE_METRICS_FILE=examples/pytest-observability/data/pytest-resource-metrics.jsonl \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=transformers-pytest-demo \
.venv/bin/python utils/configure_ci_otel.py \
  --job-name resource_demo \
  -- \
  .venv/bin/python -m pytest -p pytest_resource_monitor tests/utils/test_add_new_model_like.py -q
```

That enables:

- average CPU time per test
- average peak RSS per test
- optional CUDA memory metrics when `torch.cuda.is_available()` is true

## Build an average across multiple runs

The dashboard ranks tests by average duration across all fetched traces for the selected job. To make that meaningful, rerun the same target with the same `--job-name`.

Example:

```sh
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5317 \
OTEL_SERVICE_NAME=transformers-pytest-demo \
.venv/bin/python utils/configure_ci_otel.py \
  --job-name repeat_avg \
  -- \
  .venv/bin/python -m pytest tests/utils/test_configure_ci_otel.py -q
```

Run that command multiple times. The exporter will aggregate over recent Jaeger traces and Grafana will show the mean duration per test for `repeat_avg`.

## Grafana

Open:

- Dashboard: `http://localhost:3000/d/pytest-observability/pytest-observability`
- Explore: `http://localhost:3000/explore?orgId=1`

The main table is `Slowest Tests By Average Duration`.

It shows separate columns for:

- `Module`
- `Class`
- `Function`
- `Service`
- `Job`
- `Trace ID`
- `Avg Duration (s)`

Use the `Service` and `Job` filters at the top of the dashboard to narrow the average to a specific repeated run family.

Additional resource panels:

- `Highest Avg CPU Time`
- `Highest Avg Peak RSS`
- `Aggregated Traces`

## Jaeger

Jaeger is still useful as the raw trace UI:

- `http://localhost:16687`

You can search there by:

- service name
- `transformers.test.job`
- `transformers.test.job.id`
- `vcs.change.id` in CI runs

## Notes

- The exporter currently queries up to 200 traces over a `1h` lookback window. Those values come from `PYTEST_TRACE_EXPORTER_LIMIT` and `PYTEST_TRACE_EXPORTER_LOOKBACK` in `docker-compose.yml`.
- The average table is based on test spans only. Fixture/setup/teardown spans are not included in that ranking.
- The CPU and memory charts come from the optional pytest plugin, not from Jaeger itself.
- CUDA metrics are only emitted when the local pytest process has CUDA available.
- If you change the dashboard or exporter, reload the relevant services with:

```sh
cd examples/pytest-observability
docker compose up -d --force-recreate pytest-trace-exporter grafana
```

Use Grafana Explore with the Jaeger data source when you want to inspect an individual trace by ID or search by service and tags.
