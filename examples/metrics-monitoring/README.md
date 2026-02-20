# Metrics Monitoring

## Continuous Batching Metrics in Transformers

To setup metric monitoring with continuous batching, you will want to have tempo and prometheus running.

For this, we provide a docker compose image in `examples/metrics-monitoring`.

To run it:

```sh
cd examples/metrics-monitoring
docker compose up
```

Then, in your script running CB, you will need to create a MeterProvider and TracerProvider as follows:

```py
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create({"service.name": "transformers"})

metrics_exporter = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint="http://localhost:9090/api/v1/otlp/v1/metrics"),  # Uses OTEL_EXPORTER_OTLP_METRICS_ENDPOINT env var
    export_interval_millis=1000
)
meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_exporter])
metrics.set_meter_provider(meter_provider)

trace_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")  # Uses OTEL_EXPORTER_OTLP_TRACES_ENDPOINT env var
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
trace.set_tracer_provider(tracer_provider)
```
