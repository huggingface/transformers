# Performance (memory + speed + quantization)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Triage ladder (do these first)](#triage-ladder-do-these-first)
- [Quickstarts](#quickstarts)
  - [1) Baseline: correct device placement + mixed precision](#1-baseline-correct-device-placement--mixed-precision)
  - [2) Faster attention: set `attn_implementation`](#2-faster-attention-set-attn_implementation)
  - [3) `torch.compile`: compile once, run faster](#3-torchcompile-compile-once-run-faster)
  - [4) bitsandbytes 8-bit / 4-bit: `BitsAndBytesConfig`](#4-bitsandbytes-8-bit--4-bit-bitsandbytesconfig)
  - [5) GPTQ: post-training int4 with `gptqmodel` + `GPTQConfig`](#5-gptq-post-training-int4-with-gptqmodel--gptqconfig)
  - [6) Continuous batching for serving: `generate_batch()` / `transformers serve`](#6-continuous-batching-for-serving-generate_batch--transformers-serve)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user’s goal is **performance** in `transformers`:
- Reduce **VRAM/RAM** (fit the model)
- Increase **throughput** (tokens/sec, examples/sec)
- Reduce **latency** (time-to-first-token, p95)
- Use **quantization**, **compiled execution**, **optimized attention/kernels**, **parallelism**, or **continuous batching**

---

## Minimum questions to ask

Ask only what you need to recommend the right optimization (0–5 questions):
1) **Workload**: inference vs training? generation vs encoder-only?
2) **Target**: memory bound vs compute bound? (OOM? too slow? p95 latency? throughput?)
3) **Hardware**: CPU vs GPU (which GPU?) vs multi-GPU?
4) **Model + dtype constraints**: model id/path + `transformers` version + backend (PyTorch/TF/JAX)
5) If blocked: exact **OOM/traceback**, plus a minimal runnable snippet

---

## Triage ladder (do these first)

This ordering avoids “cool tricks” before basics:

1) **Stop accidental slow paths**
   - Batch your requests; avoid per-item loops.
   - Ensure the model and inputs are on the same device.
2) **Right-size precision**
   - Mixed precision (`float16` / `bfloat16`) usually yields large speed/memory wins on GPUs.
3) **Use an optimized attention backend**
   - Swap `attn_implementation` before changing architectures.
4) **Compile**
   - `torch.compile` can reduce Python overhead and fuse kernels.
5) **Quantize**
   - 8-bit / 4-bit (bitsandbytes) or GPTQ can be the difference between “fits” and “doesn’t”.
6) **Scale/serve**
   - Continuous batching and parallelism matter most when serving many concurrent requests.

---

## Quickstarts

### 1. Baseline: correct device placement + mixed precision

Use this when the user says “it’s slow” or “it OOMs” and you need a sane baseline.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b"  # example

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Mixed precision + automatic device placement (single GPU or multi-GPU sharding/offload)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype=torch.bfloat16,  # or torch.float16
).eval()


# Put inputs on the model's device
inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=32)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

The `dtype` argument controls the instantiated weight dtype.
- Use `dtype="auto"` to load the checkpoint’s intended dtype.
- Or force `dtype=torch.float16` / `dtype=torch.bfloat16` for mixed precision (GPU permitting).


---

### 2. Faster attention: set `attn_implementation`

Transformers exposes multiple attention backends through a single knob: `attn_implementation`.
Supported values in the attention-backends interface include (among others):  
"flash_attention_3", "flash_attention_2", "flex_attention", "sdpa" (and "eager"), plus paged variants like "paged|flash_attention_3" / "paged|flash_attention_2" / "paged|sdpa" / "paged|eager".


```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    attn_implementation="flash_attention_2",
)
```

You can also switch implementations at runtime without reloading:

```python
model.set_attn_implementation("sdpa")
```

If you don’t want to install a FlashAttention package (CUDA/PyTorch version mismatch pain), you can load a compiled kernel from the Hub via the Kernels integration:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    attn_implementation="kernels-community/flash-attn2",
)
```

**Gotchas (read before benchmarking):**
- **Backend availability depends on model + PyTorch/CUDA + dtype.**  
  For example, FlashAttention2 requires CUDA and typically `float16` or `bfloat16`; it will silently fall back or error if the dtype or build is incompatible.
- **FlashAttention2 does not support attention over padded tokens.**  
  In batched generation with padding, this can reduce performance unless you avoid padding, unpad inputs, or use an alternative backend (e.g. SDPA).
- **Some attention params force a fallback to eager.**
  For example, `output_attentions=True` is unsupported in some optimized attention paths and triggers a fallback warning.

---

### 3. `torch.compile`: static cache + compile `forward` (generation)

For generation workloads, Transformers recommends enabling StaticCache via `cache_implementation="static"`. This also turns on automatic compilation of the decoding stage for greedy and sampling decode. You can control this via `compile_config` (or disable it with `disable_compile`) and still need stable shapes to avoid recompilation.


```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    dtype="auto",  # or torch.float16 / torch.bfloat16
).eval()

# Compile the forward pass; generate() calls model.forward internally
model.forward = torch.compile(
    model.forward,
    mode="reduce-overhead",
    fullgraph=True,
)

# Keep shapes stable to avoid recompilation
inputs = tokenizer(
    "Hello!",
    return_tensors="pt",
    pad_to_multiple_of=8,
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=32, cache_implementation="static")

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Notes:
- The **first call is slower** due to compilation; benchmark after warmup.
- Keep batch size, prompt length, and `max_new_tokens` stable to avoid recompilation.
- If `fullgraph=True` fails due to graph breaks, retry with `fullgraph=False`.

---

### 4. bitsandbytes 8-bit / 4-bit: `BitsAndBytesConfig`

This is the fastest “make it fit” move for many LLMs. Install deps first:

```bash
pip install --upgrade transformers accelerate bitsandbytes
```

**8-bit example (generation path):**

```python
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",
    quantization_config=quantization_config,
)

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
```
**Preferred API:**  
Use `quantization_config=BitsAndBytesConfig(...)` when loading models.  
Avoid passing `load_in_8bit` or `load_in_4bit` directly to `from_pretrained()`; these flags exist for compatibility but are not the recommended interface.

Notes:
- The GPU performance guide explicitly recommends **using `generate()` rather than the Pipeline API** for **8-bit text generation**, because Pipeline is not optimized for 8-bit models and some sampling strategies may not be supported there.
- For multi-GPU/distributed, you can pass `max_memory={...}` to control per-device allocation when using `device_map="auto"`.

---

### 5. GPTQ: post-training int4 with `gptqmodel` + `GPTQConfig`

Transformers’ GPTQ doc states:
- GPTQ is supported via the **`gptqmodel`** package.
- Transformers supports GPTQ via GPTQModel and still documents AutoGPTQ, but AutoGPTQ is likely to be deprecated; prefer GPTQModel going forward.

Install:

```bash
pip install --upgrade accelerate optimum transformers
pip install gptqmodel --no-build-isolation
```

Quantize (example pattern):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",
    quantization_config=gptq_config,
)
```

If you hit memory pressure during quantization, GPTQ docs recommend using `max_memory={...}` (disk offloading is not supported for the dataset).

---

### 6. Continuous batching for serving: `generate_batch()` / `transformers serve`

Continuous batching increases throughput and reduces latency by dynamically re-forming the batch each step (removing finished requests and adding new ones) to avoid GPU idling. It works with `transformers serve` and `generate_batch()`.

- **PagedAttention is automatically enabled under continuous batching.**  
  You can also explicitly select a paged backend via `attn_implementation="paged|..."` if needed.


Minimal `generate_batch()` shape (tokenized inputs list + `GenerationConfig`):

```python
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    attn_implementation="sdpa_paged",
    device_map="cuda",
    dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    padding_side="left",
)

dataset = datasets.load_dataset("openai/gsm8k", "socratic", split="test").select(range(8))
tokenized = dataset.map(lambda x: tokenizer(x["question"]), batched=True)

simple_batch_inputs = [item["input_ids"] for item in tokenized]

generation_config = GenerationConfig(
    max_new_tokens=32,
    do_sample=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_batch_tokens=512,  # token budget for batching
    use_cuda_graph=False,
)

batch_outputs = model.generate_batch(inputs=simple_batch_inputs, generation_config=generation_config)

for request_id, output in batch_outputs.items():
    print(request_id, tokenizer.decode(output.generated_tokens, skip_special_tokens=True))
```

If you need custom scheduling, the docs expose a `ContinuousBatchingManager` and schedulers (default FIFO).

---

## Knobs that matter (3–8)

Prioritize these knobs before anything else:

1) **Batching + padding strategy**
   - batch requests; for LLM generation use left-padding (`padding_side="left"`) when appropriate
2) **Placement**: `device` vs `device_map` (and **inputs on `model.device`**)
3) **Precision**: `dtype` / `torch_dtype` (fp16/bf16) vs full fp32
4) **Attention backend**: `attn_implementation` (FlashAttention/SDPA/paged variants)
5) **Compilation**: `torch.compile(...)` knobs (`mode`, `fullgraph`) or compile-via-`generate()` with a static cache
6) **Quantization**: `quantization_config` (bitsandbytes 8/4-bit, GPTQ, etc.)
7) **Memory partitioning** (multi-GPU/offload): `max_memory={...}` with `device_map="auto"`
8) **Serving throughput**: continuous batching (`generate_batch()`, `max_batch_tokens`) / `transformers serve`

---

## Pitfalls & fixes

- **“It’s still slow after moving to GPU”**
  - Inputs not on GPU → ensure `tokenizer(...).to(model.device)`
  - You’re running batch_size=1 loops → batch requests; avoid Python overhead
- **“FlashAttention enabled but errors”**
  - Use the Kernels integration (`attn_implementation="kernels-community/flash-attn2"`) to avoid local build/version mismatch
  - Or fall back to `attn_implementation="sdpa"` for a safer baseline
- **“`torch.compile` made it slower”**
  - First run includes compilation; benchmark after warmup
  - Try `mode="reduce-overhead"`; avoid recompiling on shape changes (keep shapes stable)
- **“8-bit pipeline is slow / sampling not supported”**
  - For 8-bit text generation, prefer calling `model.generate()` directly (per GPU perf guide)
- **“OOM when quantizing (GPTQ)”**
  - Use `device_map="auto"` and constrain with `max_memory={...}`
  - Prefer loading an already-quantized checkpoint from the Hub when available
- **“Serving latency spikes under load”**
  - Use continuous batching to prevent GPU idle bubbles and handle ragged request lengths
  - Tune `max_batch_tokens` and request scheduling

---

## Verify / locate in repo

Repo hotspots (performance)
- **Loading / placement (`from_pretrained`, `device_map`, `max_memory`, `dtype`)**: `src/transformers/modeling_utils.py`
- **Attention backend interface (`attn_implementation`, `set_attn_implementation`)**: docs “Attention backends” + model code in `src/transformers/models/<model>/modeling_<model>.py` (where eager/SDPA/FA branches usually live)
- **KV cache internals (Static/DynamicCache)**: `src/transformers/cache_utils.py` + KV-cache docs (shows `cache_implementation="static"` + compile behavior)
- **Generation cache/config knobs (`GenerationConfig`, cache impl wiring)**: `src/transformers/generation/configuration_utils.py`
- **Core `generate()` perf paths**: `src/transformers/generation/utils.py`
- **Continuous batching (`generate_batch`)**: `src/transformers/generation/continuous_batching/continuous_api.py`
- **Quantization config objects**: `src/transformers/utils/quantization_config.py`
- **Quantizer routing (which quantizer gets picked)**: `src/transformers/quantizers/auto.py`
- **bitsandbytes glue + bnb 4bit internals**: `src/transformers/integrations/bitsandbytes.py` and `src/transformers/quantizers/quantizer_bnb_4bit.py`
- **`transformers serve` (CLI + behavior)**: docs “Serving” and implementation under `src/transformers/commands/serving.py` (shows up in tracebacks)

When uncertain, use Skill verification indexes:
- “Does this symbol/arg exist?” → `reference/generated/public_api.md`
- “Where is it implemented?” → `reference/generated/module_tree.md`

High-signal repo search keywords (grep these):
- `attn_implementation`, `set_attn_implementation`
- `torch.compile`, `cache_implementation="static"`
- `BitsAndBytesConfig`, `quantization_config`, `load_in_8bit`, `load_in_4bit`
- `GPTQConfig`, `gptqmodel`
- `generate_batch`, `ContinuousBatchingManager`, `init_continuous_batching`, `max_batch_tokens`
- `device_map`, `max_memory`