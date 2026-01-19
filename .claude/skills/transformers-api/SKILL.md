---
name: transformers-api
description: Guides coding and debugging in the Hugging Face Transformers repo. Use when questions involve transformers APIs (pipeline, AutoModel*, AutoTokenizer, Trainer, generate), repo navigation (“where is X implemented?”), performance/quantization, export/serving, or stack traces referencing transformers/ or src/transformers/.
---

# Transformers API Navigator (Claude Code)

## Purpose
This Skill is an **operating playbook** for working with the `huggingface/transformers` codebase and answering Transformers API questions **without guessing**.

It optimizes for:
- correct API choice (pipeline vs Auto* vs Trainer vs export/perf)
- fast debugging (minimal repro-first)
- accurate repo navigation (“where is X implemented?”)
- small, testable changes when modifying the repo

This file is intentionally **high-level**. Detailed breakdowns live in individual markdown files under `reference/areas/*`.

---

## When to activate
Activate this Skill if **any** of the following are true:
- The user mentions Transformers or `transformers` APIs (`pipeline`, `AutoModel*`, `AutoTokenizer`, `Trainer`, `generate`, etc.)
- They reference Transformers artifacts (`config.json`, `tokenizer.json`, `generation_config.json`, `model.safetensors`, etc.)
- They show code importing `transformers` or stack traces mentioning `transformers/` or `src/transformers/`
- They need a Transformers-specific decision (inference vs training, generation knobs, perf/quantization, export/serving)
- They ask repo questions: “where is X implemented?”, “which file owns Y?”

Do **not** activate if the request is mostly:
- Hub/Datasets usage with no `transformers` callsite, or
- **tokenizers library internals** (the separate tokenizers repo / Rust internals) with no Transformers usage.

Do activate if it’s **Transformers usage of tokenizers/processors** (route to Preprocessing).

---

## Reference entry points 

### Buckets (open exactly ONE first)
- Inference → `reference/areas/inference.md`
- Preprocessing → `reference/areas/preprocessing.md`
- Generation → `reference/areas/generation.md`
- Training / Evaluation → `reference/areas/training.md`
- Performance / Memory / Quantization → `reference/areas/performance.md`
- Export / Serving → `reference/areas/export-serving.md`
- Repo navigation / Contributing → `reference/areas/repo-contributing.md`
- Debugging / Troubleshooting → `reference/areas/troubleshooting.md`

### Verification (“don’t hallucinate”)
- Symbol/arg exists → `reference/generated/public_api.md`
- Where implemented → `reference/generated/module_tree.md`

Full repo structure is captured in: `reference/generated/module_tree.md`

### Debug template
- Minimal repro form → `templates/minimal_repro.md`

---

## Exact sequential process (always follow this order)

### Step 1 — Classify the request (pick ONE bucket)
- **Inference** (pipelines, Auto* inference)
- **Preprocessing** (tokenizers / processors)
- **Generation** (generate/decoding/chat/streaming)
- **Training / Evaluation** (Trainer, arguments, callbacks)
- **Performance / Memory / Quantization**
- **Export / Serving**
- **Repo navigation / Contributing**
- **Debugging / Troubleshooting**

### Step 2 — Ask only what’s missing (0–5 questions, only if ambiguous)
Ask only the minimum to proceed:
1) Goal/outcome in one sentence (only if unclear)
2) Modality/task (Text / Vision / Audio / Video / Multimodal) (only if relevant)
3) Model id or local path (and revision/commit if pinned) (if loading/inference/training is involved)
4) Environment: `transformers` version + backend (PyTorch/TF/JAX) + device (CPU/CUDA/MPS) (+ rough VRAM/RAM if perf matters)
5) If blocked: full stack trace + minimal repro snippet (use `templates/minimal_repro.md`)

### Step 3 — Route first (deterministic router embedded here)
Follow this router and open **exactly one** bucket file from the list above.

#### Routing rules
- If the user is blocked by an exception/traceback, regression, or wrong output → open **Troubleshooting** first  
  **unless** it is clearly a `Trainer`/training-loop failure → open **Training** first.
- If multiple buckets match, prioritize the user’s **desired outcome** over the first keyword seen.
- If still tied, use this fixed priority order:  
  **Troubleshooting > Training > Generation > Inference > Preprocessing > Performance > Export/Serving > Repo/Contributing**

#### Routing table (open exactly ONE file first)

| User intent / signal | Open this first | Common keywords / symptoms |
|---|---|---|
| Run inference / predict / use a model quickly | `reference/areas/inference.md` | `pipeline`, `AutoModelFor*`, `from_pretrained`, logits, predict, embeddings, classification, ASR/VQA/etc. |
| Preprocessing / inputs formatting (text/vision/audio/video) | `reference/areas/preprocessing.md` | `AutoTokenizer`, `AutoProcessor`, `AutoImageProcessor`, `AutoVideoProcessor`, (audio) `FeatureExtractor`, padding, truncation, transforms, normalization, resizing, sampling rate |
| Text generation / chat behavior | `reference/areas/generation.md` | `generate`, decoding, `max_new_tokens`, sampling, beams, stop tokens, streaming, chat templates |
| Fine-tuning / training / evaluation | `reference/areas/training.md` | `Trainer`, `TrainingArguments`, `train`, `evaluate`, metrics, collators, checkpoints, distributed, FSDP/DeepSpeed/Accelerate |
| Performance / memory / quantization | `reference/areas/performance.md` | VRAM/OOM, `device_map`, `torch_dtype`, fp16/bf16, attention backends, 8-bit/4-bit, bitsandbytes/GPTQ/AWQ |
| Export / serving / deployment | `reference/areas/export-serving.md` | ONNX/export, serving, batching, vLLM/TGI/SGLang, `transformers serve` (moderate-load/experimental), `transformers chat` |
| Repo navigation / contributing / “where is X implemented?” | `reference/areas/repo-contributing.md` | “where is”, “which file”, “implementation”, `src/transformers`, tests, docs, PR, add model |
| Errors, crashes, regressions, wrong outputs | `reference/areas/troubleshooting.md` | traceback, exception, mismatch, device/dtype errors, missing files, unexpected output |

#### Verification shortcuts 
Use these only when uncertain about an API/arg/behavior, or when locating code/docs:
- **Does a symbol/arg exist?** → `reference/generated/public_api.md`
- **Where is it implemented?** → `reference/generated/module_tree.md`

#### Fallback (if nothing matches)
- Open `reference/generated/public_api.md` to identify the closest public surface area.
- Then route to the nearest bucket in the table above and continue.

### Step 4 — If blocked by an error: reproduce/triage first
If the user cannot proceed due to an exception or incorrect outputs:
- prioritize minimal repro + full stack trace + versions
- classify the failure: **loading** vs **preprocessing** vs **forward/generate** vs **Trainer** vs **integration**
- apply a targeted fix + propose the smallest next diagnostic step

### Step 5 — Verify only when uncertain (never guess)
Only consult verification sources when you are unsure about a symbol/arg/behavior/default, or when locating an implementation.

Verification order:
1) `reference/generated/public_api.md` : confirms what is publicly exposed (what exists)
2) `reference/generated/module_tree.md` : finds where it lives in `src/transformers/` (where it’s implemented)
3) Fallback if needed: inspect `src/transformers/`, `docs/source/`, and/or repo search

If `reference/generated/*` looks missing or stale, **regenerate/update it before relying on it**.
If you cannot verify, say so and point to the most likely file/module to inspect next.

### Step 6 — Respond using the output contract
Every answer must include:
- **Steps** (numbered)
- **Minimal runnable snippet** (copy/paste)
- **Pitfalls & fixes** (“If X → do Y”)
- **What to change** (3–8 knobs likely to matter)

If the user is changing repo code, also include:
- exact file paths to edit
- tests to run (smallest relevant set)

---

## Repo anchors (use when needed)
- Core library: `src/transformers/`
- Tests: `tests/`
- Docs source: `docs/source/` (commonly `docs/source/en/`)
- Examples: `examples/`

When asked “where is X implemented?”:
- use `reference/generated/module_tree.md` first
- then point to exact file paths under `src/transformers/`
- include 1–3 search keywords the user can grep for

---

## Guardrails (non-negotiable)
- Do not invent APIs/args/behavior. Verify if uncertain.
- Do not propose large refactors when a small targeted change will do.
- Behavior changes should come with a test (or an explicit reason why not).
- Keep Transformers responsibilities separate from Hub/Datasets/Accelerate/PEFT unless the integration point is the blocker.