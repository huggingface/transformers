# Transformers in Your First 15 Minutes

Welcome to 🤗 Transformers!

This guide helps beginners quickly run modern AI models with minimal setup.

In this tutorial, you'll learn:
- how to install Transformers
- how to run your first text generation model
- how to chat with an instruction model
- how to generate embeddings
- how to run image classification
- basic inference optimization tips

This guide focuses on practical onboarding rather than advanced concepts.

---

# Installation

Transformers supports Python 3.10+ and PyTorch 2.4+.

Create and activate a virtual environment.

## venv

```bash
python -m venv .venv
source .venv/bin/activate
```

## uv

```bash
uv venv .venv
source .venv/bin/activate
```

Install Transformers with PyTorch support.

## pip

```bash
pip install "transformers[torch]"
```

## uv

```bash
uv pip install "transformers[torch]"
```

Verify installation:

```python
from transformers import pipeline

print("Transformers installed successfully!")
```

---

# Your First Text Generation Model

The easiest way to use Transformers is with the `pipeline` API.

```python
from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model="Qwen/Qwen2.5-1.5B"
)

result = generator(
    "The future of artificial intelligence is",
    max_new_tokens=50
)

print(result[0]["generated_text"])
```

What happens here:
1. The model is automatically downloaded from the Hugging Face Hub
2. Your text is tokenized
3. The model generates new tokens
4. Tokens are decoded back into readable text

---

# Chat with an Instruction Model

Instruction models are designed for conversations and assistant-style tasks.

```python
import torch
from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "Explain transformers in simple words."
    }
]

response = chatbot(
    messages,
    max_new_tokens=128
)

print(response[0]["generated_text"][-1]["content"])
```

## Why use `device_map="auto"`?

`device_map="auto"` automatically places model layers on available hardware:
- GPU if available
- CPU otherwise

This makes it easier to run larger models efficiently.

---

# Generate Embeddings

Embeddings convert text into numerical vectors useful for:
- semantic search
- retrieval systems
- clustering
- recommendation systems

```python
from transformers import pipeline

embedder = pipeline(
    "feature-extraction",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedder(
    "Transformers make machine learning easier."
)

print(len(embeddings[0][0]))
```

---

# Image Classification

Transformers also supports computer vision tasks.

```python
from transformers import pipeline

classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224"
)

result = classifier(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
)

print(result)
```

Example output:

```python
[
  {"label": "tabby cat", "score": 0.98}
]
```

---

# Choosing Your First Model

If you're unsure where to start, these models are beginner-friendly.

| Use Case | Recommended Model |
|---|---|
| Text generation | `Qwen/Qwen2.5-1.5B` |
| Chat assistant | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vision | `google/vit-base-patch16-224` |
| Lightweight experiments | `distilbert-base-uncased` |

Smaller models:
- download faster
- use less memory
- are easier to experiment with locally

---

# Basic Inference Optimization Tips

## Use half precision

```python
torch_dtype=torch.float16
```

This reduces GPU memory usage and can improve inference speed.

---

## Use `device_map="auto"`

```python
device_map="auto"
```

Automatically distributes model layers across available devices.

---

## Start with smaller models

Smaller models are often better for:
- learning
- debugging
- local experimentation
- low-memory environments

---

# Where to Go Next

Explore additional capabilities:
- summarization
- translation
- speech recognition
- visual question answering
- multimodal generation

Advanced guides:
- [Optimization overview](optimization_overview)
- [Pipeline tutorial](pipeline_tutorial)
- [LLM tutorial](llm_tutorial)

You can also explore over 1M+ models on the Hugging Face Hub.

Happy building with 🤗 Transformers!