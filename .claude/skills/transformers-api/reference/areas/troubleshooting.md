# Troubleshooting (errors, wrong outputs, regressions)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: classify the failure](#decision-guide-classify-the-failure)
- [Quickstarts](#quickstarts)
  - [1) Make the error actionable (logging + minimal repro)](#1-make-the-error-actionable-logging--minimal-repro)
  - [2) Firewalled / offline / “Connection error”](#2-firewalled--offline--connection-error)
  - [3) CUDA out of memory (OOM)](#3-cuda-out-of-memory-oom)
  - [4) ImportError / missing class after copy-pasting docs](#4-importerror--missing-class-after-copy-pasting-docs)
  - [5) CUDA error: device-side assert triggered](#5-cuda-error-device-side-assert-triggered)
  - [6) Silent wrong output from padding tokens (missing attention_mask)](#6-silent-wrong-output-from-padding-tokens-missing-attention_mask)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Triage flow (repeatable checklist)](#triage-flow-repeatable-checklist)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user is **blocked** (exception, crash, hang, or wrong output) while using `transformers`, or they suspect a regression.

---

## Minimum questions to ask

Ask only what you need (0–5 questions). If the user already pasted these, don’t re-ask.

1) **Exact failure**: full traceback, or “expected vs actual output”
2) **Minimal repro**: smallest runnable snippet (use `templates/minimal_repro.md`)
3) **Versions**: `transformers`, backend (`torch` / TF / JAX), Python, CUDA (if relevant)
4) **Model + revision**: model id or local path; pinned `revision`/commit if applicable
5) **Hardware**: CPU/CUDA/MPS and rough VRAM if memory/perf related

### 1-minute triage (when the user is blocked)

1) Classify the failure (download/cache, install/version, CUDA runtime, silent correctness, task mismatch)
2) Ask at most 3 missing facts (traceback, minimal repro, versions)
3) Apply one smallest fix and one next diagnostic step

---

## Decision guide: classify the failure

Classify before fixing. Most issues fall into one of these buckets:

1) **Download / cache / connectivity**
   - “Connection error… cannot find requested files in cached path”
   - hanging at model download / corporate network / firewalled machines

2) **Install / version mismatch**
   - `ImportError: cannot import name ... from transformers`
   - missing newer models/features

3) **GPU runtime / CUDA**
   - CUDA OOM
   - `device-side assert triggered`
   - dtype/device mismatch

4) **Silent correctness bugs**
   - wrong logits/hidden states with padding
   - wrong outputs due to missing masks or wrong preprocessing

5) **Auto-class / task mismatch**
   - `ValueError: Unrecognized configuration class ... for this kind of AutoModel`
   - checkpoint doesn’t support the requested task

Then apply the smallest fix + the smallest next diagnostic step.

---

## Quickstarts

### 1. Make the error actionable (logging + minimal repro)

Turn up logging and isolate to a minimal repro **before** “trying random flags”.

```python
# 1) Make transformers logs more verbose (runtime)
from transformers.utils import logging
logging.set_verbosity_debug()   # or set_verbosity_info()
logging.enable_default_handler()
logging.enable_explicit_format()

# 2) If your script is noisy, you can also:
# logging.disable_progress_bar()
```

If you can’t change code easily, use environment variables:

```bash
# More/less logging without editing code:
TRANSFORMERS_VERBOSITY=debug python your_script.py
# To suppress "advice" warnings (not errors):
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 python your_script.py
```

Now shrink to a repro:
- one model
- one input
- one forward/generate call
- print shapes/dtypes/devices right before the failure

(Use `templates/minimal_repro.md`.)

---

### 2. Firewalled / offline / “Connection error”

Symptoms: connection errors and the cache doesn’t contain the files yet, often in restricted networks.

Two reliable patterns:

**A. Pre-download the repo, then run offline**

```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    repo_type="model",
    # revision="main",  # or a tag/commit for reproducibility
)
print(local_path)
```
Note: if the model is gated or private, you must be authenticated to download files. Use `hf auth login`, or `huggingface_hub.login()`, or pass `token=...` to loading/downloading methods (including `snapshot_download()` / `from_pretrained()`).


```bash
# Avoid HTTP calls to the Hub:
HF_HUB_OFFLINE=1 python your_script.py
```

**B. Force local-only loading (no network calls)**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("./path/to/local/directory", local_files_only=True)
```

Also sanity-check cache location if you’re in containers/CI:
- Default cache location (from `HF_HUB_CACHE`) is `~/.cache/huggingface/hub`
  - Windows: `C:\Users\<username>\.cache\huggingface\hub`
- You can redirect the cache via environment variables (priority order):
  1) `HF_HUB_CACHE` ( default )
  2) `HF_HOME`
  3) `XDG_CACHE_HOME` + `/huggingface` (only if `HF_HOME` is not set)

---

### 3. CUDA out of memory (OOM)

Start with the two levers recommended in the official Transformers troubleshooting guide (training):
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to keep the overall batch size


```python
# Trainer-side (example)
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

Common additional levers : reduce inference `batch_size`, reduce `max_length` / `max_new_tokens`, and avoid returning activation-heavy outputs (like hidden states) unless needed.

---

### 4. ImportError / missing class after copy-pasting docs

Symptom example:

`ImportError: cannot import name 'SomeNewThing' from 'transformers'`

This commonly means the docs/snippet assumes a newer version of Transformers.

Fix: upgrade Transformers (and restart the runtime/kernel):

```bash
pip install --upgrade transformers
# or install from source (latest changes):
pip install git+https://github.com/huggingface/transformers
```

If the model is *very new*, verify you’re on a version that includes it, or install from source.

---

### 5. CUDA error: device-side assert triggered

This is often a vague GPU-side error. Two reliable ways to get a real traceback:

**A. Run on CPU to get a better error message**

```python
# Important: set this before any CUDA context is initialized
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # forces CPU
```

**B. Force synchronous CUDA to pinpoint the failing op**

```python
# Important: set this before the first CUDA operation
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

Once you have a real stack trace, the most common underlying causes are:
- invalid labels / out-of-range class indices (classification)
- bad token ids (negative or >= vocab size)
- shape mismatches that only surface on GPU kernels

---

### 6. Silent wrong output from padding tokens (missing attention_mask)

Symptom: outputs/logits differ for padded sequences vs the “true” unpadded sequence, without an obvious error.

Most of the time, fix by passing `attention_mask` so the model ignores padding tokens:

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

# Two sequences, second is padded with 0
input_ids = torch.tensor([
    [7592, 2057, 2097, 2393, 9611, 2115],
    [7592,    0,    0,    0,    0,    0],
])

# Correct: mask out padding
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
])

out = model(input_ids, attention_mask=attention_mask)
print(out.logits)
```

Note: tokenizers often create `attention_mask` for you when you call them, but if you bypass tokenizers and hand-craft `input_ids`, you must provide the mask yourself.

Why it’s manual: Transformers does not automatically infer `attention_mask` from padding because some models have no padding token, and some use-cases intentionally attend to padding tokens.

---

## Knobs that matter (3–8)

Prioritize these knobs before anything else:

1) **Versions**: `transformers` + backend framework version (Torch/TF/JAX)
2) **Model identity**: model id/path + pinned `revision` (reproducibility)
3) **Connectivity mode**: `HF_HUB_OFFLINE`, `local_files_only=True`, cache location env vars
4) **Device placement**: CPU vs CUDA vs MPS; single device vs sharding (`device_map`) when relevant
5) **Batch/shape**: `batch_size`, sequence length, image size, audio length
6) **Masks**: `attention_mask` (text), pixel masks where applicable
7) **Task ↔ class match**: correct `AutoModelFor*` / pipeline task for the checkpoint
8) **Logging**: `TRANSFORMERS_VERBOSITY`, explicit formatting, disable noisy progress bars

---

## Pitfalls & fixes

- **“Connection error… cannot find requested files in cached path”**
  - You’re firewalled/offline and the model isn’t cached → pre-download (`snapshot_download`) then set `HF_HUB_OFFLINE=1`, or use `local_files_only=True`.

- **ImportError for a class shown in docs**
  - You’re on an older Transformers → upgrade or install from source.

- **OOM**
  - Lower batch/length first; then route to `reference/areas/performance.md`.

- **CUDA device-side assert**
  - Run on CPU or set `CUDA_LAUNCH_BLOCKING=1` to get a real traceback; then validate label/token id ranges.

- **Wrong outputs with padding**
  - Pass `attention_mask` (especially when you create `input_ids` manually).

- **AutoModel config mismatch**
  - The checkpoint configuration cannot be mapped to the requested task head (most commonly because the checkpoint does not support that task) → load with a compatible `AutoModel*` or choose a checkpoint that supports the task.

---

## Triage flow (repeatable checklist)

Use this flow to avoid random guessing:

1) **Freeze the environment**
   - record versions + model id + revision/commit
   - re-run in a clean venv if dependency conflicts are suspected

2) **Minimize**
   - one model
   - one batch
   - one call (forward or generate)
   - print shapes/dtypes/devices right before the failure

3) **Classify**
   - download/cache vs install/version vs CUDA runtime vs silent correctness vs task mismatch

4) **Apply the smallest fix**
   - one change at a time, re-run the minimal repro

5) **Only then expand**
   - re-introduce batching, datasets, distributed, larger inputs, etc.

6) **If you suspect a regression**
   - try the same repro on a known-good version and the current version
   - pin the version in the repro so others can reproduce it

---

## Verify / locate in repo

When uncertain, use Skill verification indexes:
- “Does this symbol/arg exist?” → `reference/generated/public_api.md`
- “Where is it implemented?” → `reference/generated/module_tree.md`

Common repo hotspots (for debugging “why is this happening?”):
- Central logging utilities: `src/transformers/utils/logging.py`
- Import/version gating: `src/transformers/utils/import_utils.py`
- Model loading + weight init: `src/transformers/modeling_utils.py`
- Auto class mappings:
  - `src/transformers/models/auto/modeling_auto.py`
  - `src/transformers/models/auto/configuration_auto.py`
- Pipelines core:
  - `src/transformers/pipelines/__init__.py`
  - `src/transformers/pipelines/base.py`

If you can’t verify quickly:
- say what you *did* verify,
- name the most likely file to inspect next,
- provide 1–3 grep keywords based on the error string.