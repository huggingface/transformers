# StreamingSentimentPipeline: Complete API Reference and Usage Guide

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Configuration Guide](#configuration-guide)
6. [Protocol Adapters](#protocol-adapters)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Deployment](#deployment)

---

## Overview

The StreamingSentimentPipeline is a comprehensive, production-ready streaming sentiment analysis system that extends the Hugging Face Transformers Pipeline API with asynchronous data ingestion, event-driven architecture, and multi-protocol support. It enables real-time sentiment analysis across WebSocket, Kafka, HTTP streaming, and file-based data sources.

### Key Features

- **Multi-Protocol Support**: WebSocket, Kafka, HTTP SSE, and file-based streaming
- **Event-Driven Architecture**: Pub/sub pattern with comprehensive event handling
- **Asynchronous Processing**: Non-blocking I/O with configurable batching
- **Resilience Patterns**: Circuit breakers, retry logic, backpressure handling
- **Full Backwards Compatibility**: Drop-in replacement for transformers Pipeline
- **Production Observability**: Metrics, logging, and tracing support

### System Requirements

- Python 3.8+
- transformers >= 4.21.0 (optional, runs in standalone mode if not available)
- torch >= 1.12.0 (optional)
- asyncio support

---

## Quick Start

### Basic Usage

```python
import asyncio
from streaming_sentiment_pipeline import StreamingSentimentPipeline, StreamingConfig

async def quick_start_example():
    # Create pipeline with default sentiment model
    pipeline = StreamingSentimentPipeline(
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # Subscribe to results
    def on_result(event):
        result = event.result
        print(f"'{result.metadata.get('model_input', '')}' -> "
              f"{result.label} ({result.score:.3f})")
    
    pipeline.subscribe_result_callback(on_result)
    
    # Initialize and start
    await pipeline.initialize()
    await pipeline.start()
    
    # Process text
    await pipeline.push({"text": "I love this product!", "id": "1"})
    await pipeline.push({"text": "This is terrible.", "id": "2"})
    
    # Wait for processing
    await asyncio.sleep(2)
    await pipeline.stop()

# Run the example
asyncio.run(quick_start_example())
```

---

## Architecture

The StreamingSentimentPipeline implements a comprehensive event-driven architecture with multi-protocol support, asynchronous processing, and built-in resilience patterns.

### Core Components

1. **Pipeline Core**: Main orchestration layer
2. **Event Bus**: Pub/sub event handling
3. **Streaming Buffer**: Batch collection and management
4. **Protocol Adapters**: WebSocket, Kafka, HTTP, File
5. **Retry Manager**: Exponential backoff and retry logic
6. **Circuit Breaker**: Fault isolation and recovery

---

## API Reference

### StreamingSentimentPipeline

The main pipeline class for streaming sentiment analysis.

```python
pipeline = StreamingSentimentPipeline(
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    config=StreamingConfig(
        batch_size=16,
        window_ms=250
    )
)
```

**Key Methods**:
- `async initialize()`: Initialize the pipeline
- `async start()`: Start processing
- `async stop()`: Gracefully stop
- `async push(item)`: Push single item
- `async push_batch(items)`: Push multiple items
- `async flush()`: Force buffer flush
- `subscribe_result_callback()`: Subscribe to results
- `subscribe_error_callback()`: Subscribe to errors
- `async get_stats()`: Get pipeline metrics

---

## Configuration Guide

### StreamingConfig

```python
config = StreamingConfig(
    batch_size=32,
    window_ms=250,
    max_batch_size=128,
    ordering=OrderingMode.BEST_EFFORT,
    buffering=BufferingConfig(
        min_batch_size=8,
        max_batch_size=64,
        max_buffer_size=512,
        drop_policy=DropPolicy.NONE
    ),
    retry=RetryConfig(
        max_attempts=3,
        base_backoff_ms=100,
        circuit_breaker_policy=CircuitBreakerPolicy.ON
    )
)
```

---

## Protocol Adapters

### WebSocket Adapter

Real-time bidirectional streaming.

### Kafka Adapter

Durable message streaming with partition support.

### HTTP SSE Adapter

Server-sent events for unidirectional streaming.

### File Adapter

Batch processing from files.

---

## Usage Examples

### Event-Driven Processing

```python
async def event_example():
    pipeline = StreamingSentimentPipeline()
    
    def on_result(event):
        print(f"Result: {event.result.label}")
    
    def on_error(event):
        print(f"Error: {event.error.message}")
    
    pipeline.subscribe_result_callback(on_result)
    pipeline.subscribe_error_callback(on_error)
    
    await pipeline.initialize()
    await pipeline.start()
    
    await pipeline.push({"text": "Great product!", "id": "1"})
    await asyncio.sleep(1)
    await pipeline.stop()
```

---

## Best Practices

1. **Choose appropriate batch sizes** based on latency requirements
2. **Use event callbacks** for reactive processing
3. **Configure retry policies** for transient errors
4. **Enable circuit breakers** for fault isolation
5. **Monitor metrics** for performance tuning

---

## Performance Optimization

### Low Latency Configuration

```python
low_latency_config = StreamingConfig(
    batch_size=4,
    window_ms=100,
    ordering=OrderingMode.BEST_EFFORT
)
```

### High Throughput Configuration

```python
high_throughput_config = StreamingConfig(
    batch_size=64,
    window_ms=500,
    max_buffer_size=2048
)
```

---

## Troubleshooting

### Common Issues

1. **High Latency**: Reduce batch size, check buffer configuration
2. **Memory Issues**: Reduce max_buffer_size, implement drop policies
3. **Connection Failures**: Check adapter configurations, verify credentials
4. **Error Loops**: Review error classification, adjust retry policies

---

## FAQ

**Q: What models are supported?**
A: Any Hugging Face sentiment model, including:
- cardiffnlp/twitter-roberta-base-sentiment-latest
- distilbert-base-uncased-finetuned-sst-2-english
- nlptown/bert-base-multilingual-uncased-sentiment

**Q: Can I use multiple protocols?**
A: Yes, add multiple adapters to the same pipeline.

**Q: How do I optimize for throughput?**
A: Use larger batch sizes, longer time windows, and BEST_EFFORT ordering.

---

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "-m", "streaming_sentiment_pipeline"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-pipeline
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: pipeline
        image: sentiment-pipeline:latest
        ports:
        - containerPort: 8080
```

---

For complete documentation and examples, visit the [GitHub repository](https://github.com/huggingface/transformers).