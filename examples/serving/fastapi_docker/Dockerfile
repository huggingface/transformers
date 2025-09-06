FROM python:3.11-slim

# System deps (optional: for tokenizers, etc.)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (leverage Docker layer cache)
COPY examples/serving/fastapi_docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY examples/serving/fastapi_docker/app /app/app

# Env knobs for perf experiments
ENV NUM_THREADS=4
ENV TORCH_COMPILE=0

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
