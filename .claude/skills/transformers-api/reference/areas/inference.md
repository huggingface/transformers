# Inference (pipelines + Auto* inference)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: `pipeline()` vs manual Auto*](#decision-guide-pipeline-vs-manual-auto)
- [Quickstarts](#quickstarts)
  - [1) Pipeline: text classification (single + batch)](#1-pipeline-text-classification-single--batch)
  - [2) Pipeline: iterate a Dataset efficiently (KeyDataset)](#2-pipeline-iterate-a-dataset-efficiently-keydataset)
  - [3) Pipeline: generator input (num_workers caveat)](#3-pipeline-generator-input-num_workers-caveat)
  - [4) Pipeline: image classification (non-text example)](#4-pipeline-image-classification-non-text-example)
  - [5) Manual Auto*: classification logits (most control)](#5-manual-auto-classification-logits-most-control)
  - [6) Manual Auto*: embeddings (mean pool)](#6-manual-auto-embeddings-mean-pool)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Chunk batching (QA / zero-shot) and why it matters](#chunk-batching-qa--zero-shot-and-why-it-matters)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user wants to **run a model for inference** (predict/classify/score/encode) in `transformers`.

---

## Minimum questions to ask

Ask only what you need to produce a runnable snippet (0–5 questions):
1) **Task** (e.g., `text-classification`, `question-answering`, `automatic-speech-recognition`, `image-classification`, `feature-extraction`)
2) **Model id or local path** (and `revision` if pinned)
3) **Backend + device** (PyTorch/TF/JAX; CPU/CUDA/MPS; rough VRAM if relevant)
4) **Input modality** (text/image/audio) if unclear
5) If blocked: **full traceback + exact versions** + smallest repro

---

## Decision guide: `pipeline()` vs manual Auto*

### Prefer `pipeline()` when…
- You want the fastest path to correct inference with task-specific preprocessing/postprocessing
- You want easy batching or dataset iteration
- You’re okay with outputs formatted by the task pipeline

### Prefer manual Auto* when…
- You need direct control over tensors/logits/hidden states and custom pooling/postprocessing
- You need to debug shapes/dtypes/devices precisely
- You’re integrating into an existing service/loop and want strict control

---

## Quickstarts

### 1. Pipeline: text classification (single + batch)

```python
from transformers import pipeline

pipe = pipeline(
    task="text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0,       # GPU ordinal; use -1 for CPU
    dtype="auto",   # can also be torch.float16 / "float16" for PyTorch models
)

print(pipe("This restaurant is awesome"))
print(pipe(["Great!", "Terrible..."], batch_size=8))
```

Notes:
- For large models, prefer `device_map="auto"` over a single `device` (sharding/offload).
- If you must set `trust_remote_code=True`, pin `revision=` and treat it like running third-party code.

---

### 2. Pipeline: iterate a Dataset efficiently (KeyDataset)

Recommended for large datasets: iterate the dataset directly to avoid loading everything into memory and to avoid writing your own batching loops.

```python
import datasets
from tqdm.auto import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

pipe = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0,
)

ds = datasets.load_dataset("imdb", split="test[:200]")

# Some texts tokenize longer than the model’s max sequence length (e.g., 512), causing a size-mismatch error; truncation (and padding for batching) fixes it by enforcing a consistent max length.
for out in tqdm(pipe(KeyDataset(ds, "text"), batch_size=16, truncation=True, max_length=512, padding=True)):
    pass
```

---

### 3. Pipeline: generator input (num_workers caveat)

A generator/iterator is convenient for streaming inputs (queues/HTTP/DB), but note the caveat: with iterative generators you cannot use `num_workers > 1` for multi-process preprocessing.

```python
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0,
)

def data():
    for i in range(100):
        yield f"My example {i}"

# Caveat: because this is iterative, you cannot use num_workers > 1 to preprocess in parallel.
for out in pipe(data(), batch_size=8):
    pass
```

---

### 4. Pipeline: image classification (non-text example)

Pipelines support computer vision tasks. Inputs may be:
- an HTTP(S) URL string
- a local file path string
- a PIL image object

If you pass a *batch* of images, they must all be in the same format (all URLs, all paths, or all PIL images).

```python
from transformers import pipeline

# Vision pipelines require Pillow (PIL). If you get: "This image processor cannot be instantiated... install Pillow",
# run:  pip install -U pillow   (or: conda install -c conda-forge pillow) and restart your notebook/kernel.
clf = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=0,
    dtype="auto",
)

img_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
print(clf(img_url))
```

---

### 5. Manual Auto*: classification logits (most control)

Use this as the baseline when debugging correctness or needing raw logits.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

texts = ["I love this.", "I hate this."]
batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
batch = {k: v.to(device) for k, v in batch.items()}

with torch.inference_mode():
    logits = model(**batch).logits
    probs = logits.softmax(dim=-1)

print("probs:", probs)
print("pred:", probs.argmax(dim=-1))
```

---

### 6. Manual Auto*: embeddings (mean pool)

Use when the user wants embeddings/features (not generation).

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_id = "distilbert-base-uncased"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

texts = ["hello world", "another sentence"]
batch = tok(texts, return_tensors="pt", padding=True, truncation=True)
batch = {k: v.to(device) for k, v in batch.items()}

with torch.inference_mode():
    out = model(**batch)  # out.last_hidden_state: (B, T, H)
    mask = batch["attention_mask"].unsqueeze(-1).type_as(out.last_hidden_state)  # (B, T, 1)
    summed = (out.last_hidden_state * mask).sum(dim=1)  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1)               # (B, 1)
    emb = summed / counts                               # (B, H)

print("embeddings shape:", emb.shape)
```

If they need “sentence embeddings” in production:
- confirm pooling + normalization strategy
- validate with a small retrieval sanity check (nearest neighbors look sensible)

---

## Knobs that matter (3–8)

Prioritize these knobs before anything else:

1) **Task ↔ checkpoint compatibility**
   - Pipeline: correct `task` (or model with an embedded task)
   - Manual: correct `AutoModelFor*` class
2) **`model` + `revision`** (pin for reproducibility)
3) **Placement:** `device` vs `device_map`
4) **Precision:** `dtype` (pipeline) / `torch_dtype` (many manual loading paths)
5) **Batching:** list inputs + `batch_size` (avoid per-example loops)
6) **Tokenization:** `padding`, `truncation`, `max_length`
7) **Overrides:** `tokenizer`, `feature_extractor`, `image_processor`, `processor` (when default loading is wrong)
8) **Security/repro:** `trust_remote_code` (only if trusted) + pinned `revision`

Useful to know: the documented `pipeline()` constructor includes (among others)
`task`, `model`, `config`, `tokenizer`, `feature_extractor`, `image_processor`, `processor`, `revision`, `use_fast`, `token`,
`device`, `device_map`, `dtype='auto'`, `trust_remote_code`, and `model_kwargs`.

---

## Pitfalls & fixes

- **It’s slow**
  - You’re processing one-by-one → pass a **list** / **dataset iterator** and use `batch_size` (avoid per-example loops)
  - Batching isn’t always faster → **measure** on your hardware/model; batching is often most helpful on GPU
  - You’re on CPU → consider moving to GPU; (rule of thumb: batching on CPU often doesn’t help much)
  - Inputs are huge → set `truncation=True` and tune `max_length` (shorter `max_length` is usually faster/cheaper)

- **Wrong head / mismatched task**
  - Pipeline: ensure `task` matches the checkpoint’s intent (e.g., `"text-classification"` vs `"token-classification"`)
  - Manual: choose the correct `AutoModelFor*` (e.g., `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`)

- **Device / dtype issues**
  - Manual: move **both** the model and **all** input tensors to the same device
  - Inference best-practice: `model.eval()` (often already true after `from_pretrained`) + `torch.inference_mode()`
  - Pipeline placement: use **either** `device` **or** `device_map` (don’t set both)

- **Batching causes OOM**
  - Reduce `batch_size`; consider smaller `max_length`; handle OOM gracefully (retry with a smaller batch)
  - If lengths vary a lot, consider bucketing by length or using smaller `max_length` to stabilize memory
  - For large models, consider `device_map="auto"` (sharding/offload) and lower precision (`dtype="float16"` / `torch.float16` where supported; PyTorch backend)

---

## Chunk batching (QA / zero-shot) and why it matters

Some tasks (notably `question-answering` and `zero-shot-classification`) may require **multiple forward passes per “one” user input**.
Transformers handles this via a `ChunkPipeline` implementation so you can tune `batch_size` without manually accounting for how many
forward passes a single input triggers.

Practical implications:
- If a user reports “batch_size doesn’t behave as expected” for QA/zero-shot, check whether chunking is the cause.
- Don’t assume “1 input = 1 forward pass” for these pipelines.

---

## Verify / locate in repo

Common repo hotspots:
- Pipelines:
  - `src/transformers/pipelines/__init__.py` (factory/registry)
  - `src/transformers/pipelines/base.py` (base `Pipeline` / batching machinery)
  - `src/transformers/pipelines/*.py` (task implementations)
- Auto factories:
  - `src/transformers/models/auto/` (AutoModel/AutoConfig/AutoTokenizer mappings)
- Core loading utilities:
  - `src/transformers/modeling_utils.py`
  - `src/transformers/configuration_utils.py`