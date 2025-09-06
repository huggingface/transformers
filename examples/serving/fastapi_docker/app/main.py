import os

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import pipeline


# --- basic perf knobs you can measure for your 10% claim ---
# Use fewer OpenMP threads for CPU-bound ops; tune & record:
torch.set_num_threads(int(os.getenv("NUM_THREADS", "4")))
# If PyTorch 2.x available, compile kernels (measure effect):
try:
    torch._dynamo.config.suppress_errors = True
    torch_compile = os.getenv("TORCH_COMPILE", "0") == "1"
except Exception:
    torch_compile = False

task = os.getenv("TASK", "text-classification")
model = os.getenv("MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
device = 0 if torch.cuda.is_available() else -1

nlp = pipeline(task=task, model=model, device=device)
if torch_compile and hasattr(nlp.model, "forward"):
    nlp.model = torch.compile(nlp.model)  # measure before/after

app = FastAPI(title="Transformers FastAPI Inference", version="0.1.0")


class InferenceRequest(BaseModel):
    inputs: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "device": "cuda" if device == 0 else "cpu", "model": model, "task": task}


@app.post("/predict")
def predict(req: InferenceRequest):
    # batch predict; pipeline can take list[str]
    outputs = nlp(req.inputs)
    return {"outputs": outputs}
