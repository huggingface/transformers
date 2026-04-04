# Minimal Repro Template (Transformers)

Use this template to produce a **copy/paste runnable** repro that someone else can run and see the same issue.

## 0. One-line goal
**Goal:** <What should happen?>

## 1. What is happening (actual)
**Actual:** <What happens instead? Include exact error message or wrong output>

## 2. Environment (must be exact)
Fill in all that apply.

- OS: Windows / Linux / macOS (include version)
- Python: `python -V`
- Transformers: `python -c "import transformers; print(transformers.__version__)"`
- Backend: PyTorch / TensorFlow / JAX (pick one)
  - PyTorch: `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`
  - TF: `python -c "import tensorflow as tf; print(tf.__version__)"`
  - JAX: `python -c "import jax; print(jax.__version__)"`
- Device: CPU / CUDA / MPS
- GPU (if any): model + VRAM
- Install method:
  - pip/venv OR conda (include env name)
- Install source:
  - PyPI release OR editable install from repo (`pip install -e .`) OR specific commit/revision
- Reproducibility:
  - Does it happen every run? Y/N
  - First bad version / last good version (if known)

## 3. Installation commands (exact)
Provide the minimal set of commands someone needs to create a clean environment.

### Option A — venv + pip
```bash
python -m venv .venv
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -U pip
pip install "transformers[torch]"  # or your exact extras
```

### Option B — conda
```bash
conda create -n repro python=3.11 -y
conda activate repro
pip install -U pip
pip install transformers
```

> If you're using the repo source, replace installs with:
> `pip install -e .` (from repo root)

## 4. Minimal script (single file)

Create `repro.py` with the smallest code that still fails.
Rules:

* Use a **single model id** (or local path) and include revision if pinned
* Set seeds
* Print versions
* Avoid unrelated features (Trainer, accelerate, etc.) unless they are the bug

```python
import os
import sys
import platform
import random

def print_env():
    print("== ENV ==")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception as e:
        print("transformers import failed:", repr(e))
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch import failed:", repr(e))
    print("HF_HOME:", os.getenv("HF_HOME"))
    print("HF_HUB_CACHE:", os.getenv("HF_HUB_CACHE"))
    print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))
    print()

def set_seeds(seed=0):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def main():
    print_env()
    set_seeds(0)

    # TODO: replace with minimal failing code
    from transformers import pipeline

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    nlp = pipeline("sentiment-analysis", model=model_id)
    print(nlp("hello world"))

if __name__ == "__main__":
    main()
```

## 5. Run command + full output

Command used:
```bash
python repro.py
```

Paste the **full output** here (don't truncate).

## 6. Expected vs actual (explicit)

* **Expected:** <exact>
* **Actual:** <exact>

## 7. Smallest knobs to try (pick only relevant)

Include only the knobs that could change the failure:

* Model: different revision / different model id
* Device: CPU vs CUDA
* dtype: `torch_dtype=float16/bfloat16/float32`
* `device_map="auto"` vs explicit device
* `low_cpu_mem_usage=True/False`
* `trust_remote_code=True/False`
* Tokenization: `padding/truncation/max_length`
* Generation: `do_sample`, `temperature`, `top_p`, `num_beams`, `max_new_tokens`
* Attention backend: SDPA / flash-attn (if applicable)
* Quantization: 8-bit/4-bit settings (bitsandbytes/GPTQ/AWQ)

## 8. If it's a repo bug (for contributors)

* Suspected module/file:
  * `src/transformers/...`
* Related tests to run:
  * `python -m pytest tests/<...> -k "<pattern>"`
* Minimal patch idea:
  * <1–3 sentences>

## 9. Attachments checklist (only if needed)

* config.json / tokenizer.json / generation_config.json
* exact traceback (full)
* small input sample(s)
* exact command line flags / env vars