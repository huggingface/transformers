# Transformers Public API (Verification Guide)

## Table of Contents

1. [Definition of “Public API”](#1-definition-of-public-api)  
2. [Version Discipline](#2-version-discipline)  
3. [Mandatory Verification Workflow](#3-mandatory-verification-workflow)  
4. [Public API Surfaces (by Area)](#4-public-api-surfaces-by-area)  
   - 4.1 [Inference](#41-inference)  
   - 4.2 [Preprocessing](#42-preprocessing)  
   - 4.3 [Model Loading & Base Classes](#43-model-loading--base-classes)  
   - 4.4 [Generation](#44-generation)  
   - 4.5 [Training / Evaluation](#45-training--evaluation)  
   - 4.6 [Performance / Quantization](#46-performance--quantization)  
   - 4.7 [Export / Serving](#47-export--serving)  
5. [Deprecations & Compatibility Traps (Verify, Don’t Assume)](#5-deprecations--compatibility-traps-verify-dont-assume)  
6. [Model Artifact Files (On-Disk Reality Check)](#6-model-artifact-files-on-disk-reality-check)  
7. [Regeneration Strategy (Keep This File Correct)](#7-regeneration-strategy-keep-this-file-correct)  
8. [Minimal Repro Template (Copy/Paste)](#8-minimal-repro-template-copypaste)

---

## 1. Definition of “Public API”

An API surface in `transformers` is considered **public** if **at least one** of the following is true:

1. It is importable directly from the top-level package:
   ```python
   from transformers import X
   ```
2. It is explicitly documented in the official Hugging Face Transformers documentation (e.g., “Main classes”, “Pipelines”, “Trainer”, “Generation”).
3. It is a documented CLI, configuration file, or runtime behavior supported in the installed version.

Everything else is **implementation detail** and must not be treated as stable or user-facing.

**Explicitly non-public by default (unless docs say otherwise):**
- `transformers.models.*`
- deep imports from `transformers.generation.*` (treat as internal **unless explicitly documented as public** and/or importable from `transformers`)
- `transformers.pipelines.*` internals
- anything in `transformers.utils.*` that is not documented as public

**Production rule:**  
If you can’t 
(a) import it from `transformers` OR 
(b) find it in the official docs for the target version OR 
(c) verify it by runtime introspection, **do not present it as supported**.

---

## 2. Version Discipline

### 2.1 Pin versions (required)
For production systems, pin **all** of:
- `transformers` (exact version or exact git commit)
- backend framework (`torch` / `tensorflow` / `jax`) version
- key accelerators if used (e.g., `accelerate`, quantization libs, ONNX runtimes)

### 2.2 Record environment fingerprint (required)
Any debugging request must include:
- `transformers.__version__`
- backend + version
- device (CPU/CUDA/MPS) + CUDA version if applicable

Minimal snippet:
```python
import transformers
print("transformers:", transformers.__version__)

try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", getattr(torch.version, "cuda", None))
except Exception as e:
    print("torch not available:", repr(e))
```

---

## 3. Mandatory Verification Workflow

This is the *only* safe way to answer “does this exist?” questions.

### 3.1 Verify a top-level symbol exists
```python
import transformers

def verify_symbol(name: str) -> None:
    ok = hasattr(transformers, name)
    print(f"{name}: {'OK' if ok else 'MISSING'}")

for name in [
    "pipeline",
    "AutoTokenizer",
    "AutoModel",
    "Trainer",
    "TrainingArguments",
    "GenerationConfig",
]:
    verify_symbol(name)
```

**If missing:**
- Do not guess alternatives.
- Use discovery helpers (below), then present only what is verifiably present.

### 3.2 Verify an argument exists (inspect signature)
Never claim a kwarg exists without checking the signature in the user’s environment.

```python
import inspect
from transformers import AutoModel

sig = inspect.signature(AutoModel.from_pretrained)
print(sig)

def has_kwarg(fn, kw: str) -> bool:
    return kw in inspect.signature(fn).parameters

print("has token?", has_kwarg(AutoModel.from_pretrained, "token"))
print("has use_auth_token?", has_kwarg(AutoModel.from_pretrained, "use_auth_token"))
```

**Rule:** If the kwarg is not in the signature, do not instruct users to pass it.

### 3.3 Discover available “Auto*” and “Config” classes
Different versions ship different helpers. Discover dynamically:

```python
import transformers

def list_names(prefix: str):
    return sorted([n for n in dir(transformers) if n.startswith(prefix)])

print("Auto*:", list_names("Auto")[:80])
print("... (total)", len(list_names("Auto")))

print("*Config:", [n for n in dir(transformers) if n.endswith("Config")][:80])
```

### 3.4 Verify runtime behavior with a minimal forward / generate
A symbol can exist but still fail due to missing extras, device issues, or incompatible model files.

**Forward sanity check:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_id = "distilbert-base-uncased"  # replace
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

inputs = tok("hello world", return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
print(type(out))
```

**Generate sanity check (only for causal/seq2seq models):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "gpt2"  # replace
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tok("Hello", return_tensors="pt")
with torch.no_grad():
    ids = model.generate(**inputs, max_new_tokens=10)
print(tok.decode(ids[0], skip_special_tokens=True))
```

---

## 4. Public API Surfaces (by Area)

**Important:** The lists below are “common public entry points”, not a guarantee for every version.  
Always run [Section 3](#3-mandatory-verification-workflow) in the user’s environment.

### 4.1 Inference

**Canonical entry point**
```python
from transformers import pipeline
```

**Verify supported tasks in the install**
```python
# Verify supported pipeline tasks WITHOUT assuming a specific registry constant exists.
from transformers import pipelines

# Prefer the documented registry if present (custom pipeline docs point to PIPELINE_REGISTRY),
# but fall back gracefully if the installed version uses something else.
if hasattr(pipelines, "PIPELINE_REGISTRY"):
    reg = pipelines.PIPELINE_REGISTRY

    # Try a few common ways a registry might expose tasks, but only use what actually exists.
    for cand in ["get_supported_tasks", "supported_tasks", "SUPPORTED_TASKS"]:
        if hasattr(reg, cand):
            obj = getattr(reg, cand)
            tasks = obj() if callable(obj) else obj
            print("num tasks:", len(tasks))
            print("example tasks:", sorted(list(tasks))[:30])
            break
    else:
        print("PIPELINE_REGISTRY present; inspect it for task listing:", [n for n in dir(reg) if "task" in n.lower()])

elif hasattr(pipelines, "SUPPORTED_TASKS"):
    tasks = pipelines.SUPPORTED_TASKS
    print("num tasks:", len(tasks))
    print("example tasks:", sorted(tasks.keys())[:30])

else:
    print("No known pipeline task registry found; inspect transformers.pipelines:", [n for n in dir(pipelines) if "task" in n.lower()])
```

**Pitfalls & fixes**
- If a pipeline task errors with “unknown task”: list `SUPPORTED_TASKS` and pick an available task name.
- If the pipeline tries to download unexpected files: confirm model id/path + revision, and verify local directory contents.

**Knobs likely to matter**
- `device` / `device_map`
- `dtype` (or `torch_dtype` in older installs , **inspect `inspect.signature(transformers.pipeline)`** before recommending)
- `batch_size`
- `max_length` / `truncation` / `padding` (varies by pipeline)
- model-specific kwargs (must be verified)

---

### 4.2 Preprocessing

**Canonical entry points**
```python
from transformers import AutoTokenizer, AutoProcessor
```

Depending on modality and version, these may or may not exist:
- `AutoImageProcessor`
- `AutoFeatureExtractor`
- `AutoVideoProcessor`

**Verify availability**
```python
import transformers
for name in ["AutoTokenizer", "AutoProcessor", "AutoImageProcessor", "AutoFeatureExtractor", "AutoVideoProcessor"]:
    print(name, hasattr(transformers, name))
```

**Pitfalls & fixes**
- “Tokenizer class not found”: verify model repo contains tokenizer artifacts (see Section 6) and that you’re using the right Auto* loader.
- “Padding/truncation mismatch”: set `padding=True/False`, `truncation=True/False`, and confirm expected tensor shapes.

**Knobs likely to matter**
- `padding`, `truncation`, `max_length`
- `return_tensors` (`"pt"`, `"tf"`, `"np"`)
- modality-specific preprocessing params (verify via processor docs or runtime inspection)

---

### 4.3 Model Loading & Base Classes

**Canonical entry points**
```python
from transformers import AutoConfig, AutoModel
```

Task-specific autos typically exist as `AutoModelFor*` classes, but do not assume which ones.
Discover in the user’s environment:

```python
import transformers
heads = sorted([n for n in dir(transformers) if n.startswith("AutoModelFor")])
print("AutoModelFor* count:", len(heads))
print("sample:", heads[:40])
```

**Base classes (commonly public)**
```python
from transformers import PreTrainedModel, PreTrainedConfig
```

**Pitfalls & fixes**
- “Unrecognized model type”: verify `config.json` has `model_type`, and that the installed `transformers` supports it.
- “Missing weights”: confirm `model.safetensors` / shards exist and match index file if sharded.

**Knobs likely to matter (verify before recommending)**
- `dtype` (or `torch_dtype` in older installs — **inspect signatures** because dtype/precision knobs vary by version/backend)
- `device_map`
- `low_cpu_mem_usage`
- auth kwargs (e.g., `token` vs older names) — verify via signature
- `trust_remote_code` (security-sensitive; do not recommend unless necessary and understood)

---

### 4.4 Generation

**Canonical surface**
- `model.generate(...)` (method on generation-capable model classes)

**Generation config (often public, verify)**
```python
import transformers
print("GenerationConfig present?", hasattr(transformers, "GenerationConfig"))
```

**Streaming helpers (often public, verify)**
```python
import transformers
for name in ["TextStreamer", "TextIteratorStreamer"]:
    print(name, hasattr(transformers, name))
```

**Pitfalls & fixes**
- “generate() got unexpected keyword”: inspect `generate` signature and/or use `model.generation_config` to set fields.
- “Stops too late / never stops”: verify EOS token id(s) and stopping criteria; confirm tokenizer special tokens.

**Knobs likely to matter**
- `max_new_tokens`, `min_new_tokens`
- `do_sample`, `temperature`, `top_p`, `top_k`
- `num_beams`, `early_stopping`
- `repetition_penalty`, `no_repeat_ngram_size`
- `eos_token_id`, `pad_token_id`
*(All must be version-verified.)*

---

### 4.5 Training / Evaluation

**Canonical Trainer surface (verify)**
```python
from transformers import Trainer, TrainingArguments
```

Optional trainer variants may exist (verify):
- `Seq2SeqTrainer`
- `Seq2SeqTrainingArguments`

**Verify availability**
```python
import transformers
for name in ["Trainer", "TrainingArguments", "Seq2SeqTrainer", "Seq2SeqTrainingArguments"]:
    print(name, hasattr(transformers, name))
```

**Pitfalls & fixes**
- “KeyError in metrics / labels”: confirm dataset fields and data collator output keys.
- “Distributed mismatch”: confirm versions of `accelerate`/backend and consistent launch method.

**Knobs likely to matter**
- `per_device_train_batch_size`, `gradient_accumulation_steps`
- `learning_rate`, `warmup_steps`, `lr_scheduler_type`
- `fp16` / `bf16` (verify supported in the version/backend)
- `logging_steps`, `eval_steps`, `save_steps`
- `report_to` integrations (verify installed extras)

---

### 4.6 Performance / Quantization

Quantization support changes across versions and depends on optional dependencies.
Never claim a quantization config exists without verifying importability.

**Discovery pattern**
```python
import transformers
candidates = [
    "BitsAndBytesConfig",
    "GPTQConfig",
    "AwqConfig",
    "QuantoConfig",
]
for name in candidates:
    print(name, hasattr(transformers, name))
```

**Pitfalls & fixes**
- “ModuleNotFoundError for quantization backend”: install required dependency and re-verify.
- “dtype/device mismatch”: ensure model weights + inputs on same device; validate `torch_dtype`.

**Knobs likely to matter**
- `device_map`
- `dtype` (or `torch_dtype` in older installs — **inspect signatures** because dtype/precision knobs vary by version/backend)
- quantization config object fields (version-dependent; verify via signature/dir)

---

### 4.7 Export / Serving

Export/serving is often handled by adjacent tooling (e.g., ONNX/export toolchains and serving runtimes).
Do not invent “native exporter APIs” unless you verify they exist in the target version and are documented.

**Safe guidance approach**
1. Identify the target runtime (ONNX Runtime / TensorRT / TGI / vLLM / etc.).
2. Verify which tool owns export in the user’s stack (Transformers vs external).
3. Provide only documented + verifiable steps.

**Pitfalls & fixes**
- “Export fails due to unsupported ops”: confirm opset, model architecture, and runtime support.

---

## 5. Deprecations & Compatibility Traps (Verify, Don’t Assume)

This section is intentionally conservative: it tells you **how** to verify, not **what** to assume.

### 5.1 Authentication keyword arguments
Auth-related kwargs have changed over time across the ecosystem.
**Always inspect `from_pretrained` signature**:
```python
import inspect
from transformers import AutoTokenizer
print(inspect.signature(AutoTokenizer.from_pretrained))
```
Only recommend kwargs that appear in the signature.

### 5.2 Download/cache kwargs
Download/caching controls can change; some kwargs become no-ops or get removed.
Again: inspect signatures and/or consult official docs for the pinned version.

### 5.3 “Internal helpers” are not stable
If a solution requires importing from deep modules (e.g., `transformers.models...`), treat it as:
- “implementation detail”
- “may break across versions”
- “should be avoided unless you own the pinned commit”

---

## 6. Model Artifact Files (On-Disk Reality Check)

These are common files found in HF model repos or local export directories; actual sets vary.

**Common config/tokenizer files**
- `config.json`
- `generation_config.json` (may be absent)
- `tokenizer.json` (fast tokenizer)
- `tokenizer_config.json`
- `special_tokens_map.json`

**Common weights files**
- `model.safetensors` (or sharded: `model-00001-of-000xx.safetensors` + index json)
- `pytorch_model.bin` (legacy)
- backend-specific equivalents may exist depending on framework

**Sanity check: load config + tokenizer**
```python
from transformers import AutoConfig, AutoTokenizer

path_or_id = "YOUR_MODEL"  # local path or model id
cfg = AutoConfig.from_pretrained(path_or_id)
tok = AutoTokenizer.from_pretrained(path_or_id)

print("model_type:", getattr(cfg, "model_type", None))
print("tokenizer:", tok.__class__.__name__)
```

**If load fails**
- Confirm the directory contains expected artifacts.
- Confirm backend compatibility (Torch vs TF vs Flax).
- If `trust_remote_code` is involved, treat it as a security decision:
  - verify it is required
  - verify the exact repo revision you trust

---

## 7. Regeneration Strategy (Keep This File Correct)

This file should remain correct across releases by being **workflow-first** and **snapshot-driven**, not a giant hardcoded list.

### 7.1 CI snapshot (recommended)
In your pinned environment, run a script that records:
- `transformers.__version__`
- top-level symbols (filtered)
- available `Auto*` classes
- available quantization config candidates

Example snapshot script:
```python
import json
import transformers

def filt(names, prefixes=(), suffixes=(), contains=()):
    out = []
    for n in names:
        if prefixes and not any(n.startswith(p) for p in prefixes):
            continue
        if suffixes and not any(n.endswith(s) for s in suffixes):
            continue
        if contains and not any(c in n for c in contains):
            continue
        out.append(n)
    return sorted(out)

names = dir(transformers)
snapshot = {
    "transformers_version": transformers.__version__,
    "top_level_selected": filt(
        names,
        prefixes=("Auto", "PreTrained", "Text", "Trainer", "Training", "Generation", "pipeline"),
        suffixes=(),
        contains=("Config",),
    )[:2000],
    "auto_classes": filt(names, prefixes=("Auto",)),
    "model_for_heads": sorted([n for n in names if n.startswith("AutoModelFor")]),
    "config_like": sorted([n for n in names if n.endswith("Config")]),
}

print(json.dumps(snapshot, indent=2)[:20000])
```

Store this snapshot alongside releases and update this file if:
- major surfaces change
- verification steps need to accommodate new patterns

### 7.2 What never changes
Even when symbols change, the safe workflow remains:
- check importability
- inspect signatures
- run minimal repro

---

## 8. Minimal Repro Template (Copy/Paste)

Use this when users report errors. Require them to fill it.

```python
"""
MINIMAL REPRO TEMPLATE (Transformers)

1) Environment
- transformers==?
- backend: torch/tf/jax == ?
- device: CPU/CUDA/MPS (+ CUDA version if relevant)
- OS: ?

2) Model
- model id or local path:
- revision/commit (if pinned):
- trust_remote_code: True/False (and why)

3) Repro
- exact code below
- exact traceback output
"""

import transformers
print("transformers:", transformers.__version__)

# Optional backend info
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", getattr(torch.version, "cuda", None))
except Exception as e:
    print("torch not available:", repr(e))

MODEL = "REPLACE_ME"

# Choose one path (tokenizer/model OR pipeline) depending on issue:
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

inputs = tok("hello", return_tensors="pt")
out = model(**inputs)
print(type(out))
```

---