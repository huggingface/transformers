# FastAPI + Docker Deployment for Transformers Inference

This example demonstrates how to deploy Hugging Face Transformers models as a production-ready FastAPI service with Docker containerization. The service provides a RESTful API for any Transformers pipeline with configurable performance optimizations.

## What it is

A FastAPI inference service that:
- Supports any Hugging Face Transformers pipeline (text classification, text generation, NER, etc.)
- Provides automatic batching for improved throughput
- Includes performance optimizations (thread control, optional torch.compile)
- Offers Docker containerization for easy deployment
- Includes health checks and proper error handling

## Quick Start

### Run Locally

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn transformers[torch]
   ```

2. **Start the server:**
   ```bash
   cd examples/serving/fastapi_docker
   uvicorn app.main:app --reload --port 8000
   ```

3. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Make predictions
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs":["I love this!", "This is terrible."]}'
   ```

### Run with Docker

1. **Build the image:**
   ```bash
   docker build -t hf-fastapi:latest -f examples/serving/fastapi_docker/Dockerfile .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 hf-fastapi:latest
   ```

3. **Test the containerized service:**
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"inputs":["I love this!", "This is terrible."]}'
   ```

## Configuration

### Environment Variables

Configure the service using these environment variables:

- `TASK`: Pipeline task (default: `text-classification`)
- `MODEL`: Model name or path (default: `distilbert-base-uncased-finetuned-sst-2-english`)
- `NUM_THREADS`: OpenMP threads for CPU inference (default: `4`)
- `TORCH_COMPILE`: Enable PyTorch 2.x compilation (default: `0`, set to `1` to enable)

### Examples

**Text Classification (default):**
```bash
docker run -p 8000:8000 \
  -e TASK=text-classification \
  -e MODEL=distilbert-base-uncased-finetuned-sst-2-english \
  hf-fastapi:latest
```

**Text Generation:**
```bash
docker run -p 8000:8000 \
  -e TASK=text-generation \
  -e MODEL=gpt2 \
  hf-fastapi:latest
```

**Named Entity Recognition:**
```bash
docker run -p 8000:8000 \
  -e TASK=ner \
  -e MODEL=dbmdz/bert-large-cased-finetuned-conll03-english \
  hf-fastapi:latest
```

## Performance Optimizations

This example includes several configurable performance optimizations:

### 1. Thread Control
- Configurable OpenMP threads via `NUM_THREADS`
- Optimal setting depends on your CPU and workload
- Test different values (2, 4, 8) and measure throughput

### 2. PyTorch Compilation (PyTorch 2.x)
- Enable with `TORCH_COMPILE=1`
- Compiles model for faster inference
- May have longer startup time but faster subsequent requests

### 3. Automatic Batching
- Send multiple inputs in a single request
- Better GPU utilization and higher throughput
- Example: `{"inputs": ["text1", "text2", "text3", ...]}`

### Performance Measurement

To measure improvements, benchmark before and after applying optimizations:

```bash
# Baseline
time curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs":["Sample text"] * 32}'

# With optimizations
NUM_THREADS=8 TORCH_COMPILE=1 uvicorn app.main:app --port 8001 &
time curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs":["Sample text"] * 32}'
```

## API Reference

### Endpoints

#### `GET /health`
Returns service health and configuration.

**Response:**
```json
{
  "status": "ok",
  "device": "cpu",
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "task": "text-classification"
}
```

#### `POST /predict`
Perform inference on input texts.

**Request:**
```json
{
  "inputs": ["Text to classify", "Another text"]
}
```

**Response:**
```json
{
  "outputs": [
    [{"label": "POSITIVE", "score": 0.9998}],
    [{"label": "NEGATIVE", "score": 0.9995}]
  ]
}
```

## Testing

Run the test suite:

```bash
cd examples/serving/fastapi_docker
pip install pytest requests
pytest tests/test_api.py -v
```

## Security Considerations

⚠️ **Important Security Notes:**

- This example is for demonstration purposes
- Do not expose this service directly to the internet without proper authentication
- Consider using API keys, rate limiting, and input validation for production
- Review Hugging Face's [security guidelines](https://huggingface.co/docs/hub/security) for model serving

## Deployment Options

### Docker Compose
```yaml
version: '3.8'
services:
  transformers-api:
    build:
      context: .
      dockerfile: examples/serving/fastapi_docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - TASK=text-classification
      - MODEL=distilbert-base-uncased-finetuned-sst-2-english
      - NUM_THREADS=4
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformers-fastapi
spec:
  replicas: 3
  selector:
    matchLabels:
      app: transformers-fastapi
  template:
    metadata:
      labels:
        app: transformers-fastapi
    spec:
      containers:
      - name: api
        image: hf-fastapi:latest
        ports:
        - containerPort: 8000
        env:
        - name: NUM_THREADS
          value: "4"
```

## Contributing

When contributing improvements:
1. Test performance changes with benchmarks
2. Update documentation with new configuration options
3. Ensure Docker builds pass
4. Add tests for new features

## License

This example follows the same license as the Hugging Face Transformers library.
