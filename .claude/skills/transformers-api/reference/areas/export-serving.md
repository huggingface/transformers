# Export & Serving (deployment, runtimes, CLIs)

## Contents
- [Scope](#scope)
- [Minimum questions to ask](#minimum-questions-to-ask)
- [Decision guide: serve in Python vs export a portable artifact](#decision-guide-serve-in-python-vs-export-a-portable-artifact)
- [Quickstarts](#quickstarts)
  - [1) Local OpenAI-compatible server (`transformers serve`)](#1-local-openai-compatible-server-transformers-serve)
  - [2) Sanity-check the server (curl)](#2-sanity-check-the-server-curl)
  - [3) Export to ONNX (Optimum CLI)](#3-export-to-onnx-optimum-cli)
  - [4) Load + run an ONNX export (ORTModel)](#4-load--run-an-onnx-export-ortmodel)
  - [5) Export to ExecuTorch (edge/mobile)](#5-export-to-executorch-edgemobile)
  - [6) Export to TorchScript (PyTorch-only; limited)](#6-export-to-torchscript-pytorch-only-limited)
- [Knobs that matter (3–8)](#knobs-that-matter-38)
- [Pitfalls & fixes](#pitfalls--fixes)
- [Verify / locate in repo](#verify--locate-in-repo)

---

## Scope

Use this page when the user wants to **ship** a Transformers model:
- serve it behind an HTTP API (local dev → deployment)
- export it to another runtime (ONNX / ExecuTorch / TFLite via Optimum; TorchScript via PyTorch/Transformers)
- choose the right “packaging” path given constraints (latency/throughput, hardware, Python vs non-Python)

---

## Minimum questions to ask

Ask only what you need to pick a path (0–5 questions):
1) **Workload**: encoder inference (cls/embeddings) vs **LLM generation** (chat/completions)?
2) **Target runtime**: must run **outside Python**? must run on **mobile/edge**? OpenAI-compatible API required?
3) **Hardware**: CPU / CUDA GPU / MPS / edge accelerator; memory limits
4) **Model id/path + revision** (pin if you care about reproducibility)
5) If blocked: exact error + smallest repro + versions (`transformers`, PyTorch, CUDA, Optimum/runtime)

---

## Decision guide: serve in Python vs export a portable artifact

### Choose “Serve” when…
- you want a fast integration path for an app
- you can keep Python in the stack
- you want an HTTP boundary (and potentially OpenAI-compatible endpoints)

Typical choices:
- **`transformers serve`**: quick local server; good for dev/moderate load
- production LLM throughput: consider dedicated serving stacks (outside this repo) that specialize in continuous batching, KV cache, tensor parallel, etc.

### Choose “Export” when…
- you must run in a non-Python runtime
- you need a portable artifact for inference engines / mobile / embedded

Typical choices:
- **ONNX** (via Optimum): broad runtime support
- **ExecuTorch** (via Optimum): PyTorch-native edge/mobile packaging
- **TorchScript**: PyTorch-only and can be brittle; best for simpler encoder models
- **TFLite** (via Optimum TF exporters): TensorFlow Lite ecosystems (mobile/edge), often needs fixed shapes

---

## Quickstarts

### 1. Local OpenAI-compatible server (`transformers serve`)

Use this for local/dev integration tests. Always check the current flags in your environment:

```bash
transformers serve --help
```

Install serving dependencies:

```bash
pip install transformers[serving]
```

Then start the server:

```bash
transformers serve 
```
# Optional: force a single model for all requests (avoids per-request model hints)
# transformers serve --force-model "Qwen/Qwen2.5-0.5B-Instruct"

Notes:
- Treat this as **developer-friendly** serving. For high-QPS production, you’ll usually reach for specialized serving runtimes.

---

### 2. Sanity-check the server (curl)

Chat Completions request (OpenAI-compatible):

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"system","content":"hello"}],"temperature":0.9,"max_tokens":1000,"stream":true,"model":"Qwen/Qwen2.5-0.5B-Instruct"}'
```

The same server also supports the Responses API:

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "stream": true,
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
```

If requests fail:
- run `transformers serve --help` and confirm host/port/model settings
- confirm the client `model` string matches what the server expects

---

### 3. Export to ONNX (Optimum CLI)

Install Optimum ONNX tooling:

```bash
pip install optimum-onnx
```

Export a model to ONNX:

```bash
optimum-cli export onnx \
  --model distilbert/distilbert-base-uncased-distilled-squad \
  distilbert_squad_onnx/
```

Notes:
- If exporting from a local directory, ensure tokenizer/config live alongside weights.
- If task inference is ambiguous, pass `--task` (e.g., `question-answering`, `text-classification`, `text-generation`).

---

### 4. Load + run an ONNX export (ORTModel)

```python
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering

onnx_dir = "distilbert_squad_onnx"

tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
model = ORTModelForQuestionAnswering.from_pretrained(onnx_dir)

inputs = tokenizer(
    "What runtime is this?",
    "This is ONNX Runtime via Optimum.",
    return_tensors="pt",
)
outputs = model(**inputs)

print(outputs.start_logits.shape, outputs.end_logits.shape)
```

Sanity validation tip:
- compare logits on 3–10 fixed inputs between PyTorch and ONNX before shipping

---

### 5. Export to ExecuTorch (edge/mobile)

This is a practical path when you want a PyTorch-native on-device artifact.

Install ExecuTorch exporter dependencies:

```bash
git clone https://github.com/huggingface/optimum-executorch.git
cd optimum-executorch
pip install .
```

Export (CLI):

```bash
optimum-cli export executorch \
  --model "HuggingFaceTB/SmolLM2-135M-Instruct" \
  --task "text-generation" \
  --recipe "xnnpack" \
  --output_dir "smollm2_executorch"
```

Run (Python wrapper around the exported artifact):

```python
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = ExecuTorchModelForCausalLM.from_pretrained("smollm2_executorch/")

prompt = "Explain KV cache in one sentence."
print(model.text_generation(tokenizer=tokenizer, prompt=prompt, max_seq_len=64))
```

Validation tip:
- run the same 3–10 prompts on the original model and the exported artifact; compare outputs at the token level where possible (or at least consistent decoding settings)

---

### 6. Export to TorchScript (PyTorch-only; limited)

TorchScript is best for simpler, stable encoder-style graphs. Many Transformers models require enabling TorchScript mode so outputs are traceable.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    torchscript=True,  # important for many models
).eval()

# Dummy inputs for tracing (trace will specialize to these shapes)
ex = tok("hello world", return_tensors="pt")

with torch.no_grad():
    traced = torch.jit.trace(model, (ex["input_ids"], ex["attention_mask"]))

traced.save("model_ts.pt")
```

Pitfalls:
- `torchscript=True` is required for models with tied weights (typically models with a language-model head). Models without an LM head can be exported without it.
- tracing is shape-sensitive; the trace generally only supports the same input shapes used during tracing (pad/choose a max expected shape).


---

## Knobs that matter (3–8)

1) **Serve vs export**
   - Need an API quickly → serve
   - Need a portable artifact / non-Python runtime → export
2) **Workload**
   - LLM generation is sensitive to KV-cache + batching; encoder inference exports more easily
3) **Repro pinning**
   - pin model `revision` and record tool/runtime versions
4) **Export “task”**
   - pass `--task` when exporting local models or ambiguous checkpoints
5) **Shapes**
   - TorchScript and many mobile exports are sensitive to shapes; validate with representative inputs
6) **Runtime choice**
   - ONNX Runtime vs other accelerators; for edge/mobile consider ExecuTorch/TFLite
7) **Correctness validation**
   - always compare outputs on a small fixed suite before shipping
8) **Performance validation**
   - measure latency/throughput on the target hardware (not just dev machine)

---

## Pitfalls & fixes

- **Server starts but requests fail**
  - check `transformers serve --help` for port/model routing
  - confirm endpoint path and request JSON match what your server expects
- **ONNX export “works” but outputs differ**
  - verify tokenizer parity (same files/config), and compare logits first
  - ensure you didn’t accidentally change padding/truncation/max_length
- **TorchScript breaks on real inputs**
  - tracing used one example shape; real shapes differ → prefer ONNX or constrain shapes
- **Edge export slow**
  - ensure you chose an appropriate recipe/backend and validated quantization/perf settings for the device

---

## Verify / locate in repo

Use Skill verification indexes when uncertain:
- “Does this symbol/arg exist?” → `reference/generated/public_api.md`
- “Where is it implemented?” → `reference/generated/module_tree.md`

Useful repo grep keywords:
- `transformers serve`, `openai`, `chat/completions`, `responses`
- `export`, `onnx`, `executorch`, `torchscript`
- `pipelines`, `generation`, `cache`, `continuous batching` (if serving overlaps with perf questions)