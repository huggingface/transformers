import torch
from transformers import pipeline

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

pipe = pipeline(
    "sentiment-analysis",
    device=0 if device == "mps" else -1
)

print(f"Using device: {device}")
print(pipe("Transformers on Apple Silicon is awesome!"))
