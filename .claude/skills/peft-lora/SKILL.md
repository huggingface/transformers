---
name: peft-lora
description: Guides efficient fine-tuning with PEFT (Parameter-Efficient Fine-Tuning) methods, especially LoRA and QLoRA. Use when questions involve LoRA adapters, QLoRA quantized fine-tuning, PEFT configurations (LoraConfig, r, lora_alpha), adapter loading/merging, memory-efficient training, or troubleshooting PEFT-related issues.
---

# PEFT & LoRA Navigator (Claude Code)

## Purpose
This Skill is an **operating playbook** for working with Parameter-Efficient Fine-Tuning (PEFT), focusing on LoRA (Low-Rank Adaptation) and QLoRA methods within the Hugging Face ecosystem.

It optimizes for:
- correct PEFT configuration (LoraConfig parameters, target modules)
- memory-efficient fine-tuning (QLoRA, 4-bit/8-bit quantization)
- adapter management (loading, merging, stacking)
- troubleshooting common PEFT pitfalls
- integration with Transformers Trainer

This file is intentionally **high-level**. Detailed breakdowns live in individual markdown files under `reference/areas/*`.

---

## When to activate
Activate this Skill if **any** of the following are true:
- The user mentions PEFT, LoRA, QLoRA, adapters, or parameter-efficient fine-tuning
- They reference PEFT artifacts (`adapter_config.json`, `adapter_model.safetensors`, `lora_config`)
- They show code importing `peft` or stack traces mentioning `peft/`
- They need memory-efficient training decisions (LoRA vs full fine-tuning vs QLoRA)
- They ask about adapter operations: merge, save, load, switch, combine
- They mention `LoraConfig`, `r`, `lora_alpha`, `target_modules`, `modules_to_save`
- They want to fine-tune large LLMs on consumer hardware

Do **not** activate if the request is mostly:
- Full fine-tuning without PEFT, or
- Hub/Datasets usage with no PEFT/adapter context, or
- General Transformers inference without adapters

Do activate if it's **PEFT integration with Transformers** (training/inference with adapters).

---

## Reference entry points 

### Buckets (open exactly ONE first)
- LoRA Basics → `reference/areas/lora-basics.md`
- QLoRA & Quantization → `reference/areas/qlora-quantization.md`
- Adapter Operations → `reference/areas/adapter-operations.md`
- Training Integration → `reference/areas/training-integration.md`
- Troubleshooting → `reference/areas/troubleshooting.md`

### Debug template
- Minimal repro form → `templates/minimal_repro.md`

---

## Exact sequential process (always follow this order)

### Step 1 — Classify the request (pick ONE bucket)
- **LoRA Basics** (LoraConfig, r, lora_alpha, target_modules, getting started)
- **QLoRA & Quantization** (4-bit/8-bit, BitsAndBytesConfig, memory-constrained training)
- **Adapter Operations** (load, save, merge, switch, combine adapters)
- **Training Integration** (Trainer with PEFT, SFTTrainer, custom loops)
- **Troubleshooting** (dtype issues, loading errors, wrong outputs)

### Step 2 — Ask only what's missing (0–5 questions, only if ambiguous)
Ask only the minimum to proceed:
1) Goal/outcome in one sentence (only if unclear)
2) Model id or local path (and revision if pinned)
3) PEFT method (LoRA / QLoRA / other) and config (r, lora_alpha, target_modules)
4) Environment: `peft` version + `transformers` version + device (CPU/CUDA/MPS) + VRAM
5) If blocked: full stack trace + minimal repro snippet (use `templates/minimal_repro.md`)

### Step 3 — Route first (deterministic router embedded here)
Follow this router and open **exactly one** bucket file from the list above.

#### Routing rules
- If the user is blocked by an exception/traceback → open **Troubleshooting** first
- If multiple buckets match, prioritize the user's **desired outcome** over the first keyword seen.
- If still tied, use this fixed priority order:
  **Troubleshooting > Training Integration > QLoRA & Quantization > LoRA Basics > Adapter Operations**

#### Routing table (open exactly ONE file first)

| User intent / signal | Open this first | Common keywords / symptoms |
|---|---|---|
| New to LoRA / setting up first adapter | `reference/areas/lora-basics.md` | `LoraConfig`, `r`, `lora_alpha`, `target_modules`, `get_peft_model`, first time, basics |
| Memory-constrained / 4-bit / 8-bit training | `reference/areas/qlora-quantization.md` | QLoRA, `BitsAndBytesConfig`, `load_in_4bit`, quantization, OOM, consumer GPU |
| Loading / saving / merging adapters | `reference/areas/adapter-operations.md` | `merge_and_unload`, `save_pretrained`, `PeftModel.from_pretrained`, multiple adapters |
| Training with Trainer / SFTTrainer | `reference/areas/training-integration.md` | `Trainer`, `SFTTrainer`, `prepare_model_for_kbit_training`, training loop |
| Errors, crashes, wrong outputs | `reference/areas/troubleshooting.md` | traceback, dtype, mismatch, `ValueError`, loading fails, outputs wrong |

#### Fallback (if nothing matches)
- Open `reference/areas/lora-basics.md` to establish fundamentals
- Then route to the nearest bucket in the table above and continue

### Step 4 — If blocked by an error: reproduce/triage first
If the user cannot proceed due to an exception or incorrect outputs:
- prioritize minimal repro + full stack trace + versions
- classify the failure: **config** vs **loading** vs **training** vs **dtype** vs **merging**
- apply a targeted fix + propose the smallest next diagnostic step

### Step 5 — Verify only when uncertain (never guess)
Only consult documentation when you are unsure about a parameter/behavior/default.

Verification order:
1) PEFT documentation: https://huggingface.co/docs/peft
2) Transformers PEFT integration docs
3) Fallback if needed: inspect `peft` source or repo search

If you cannot verify, say so and point to the most likely source to inspect next.

### Step 6 — Respond using the output contract
Every answer must include:
- **Steps** (numbered)
- **Minimal runnable snippet** (copy/paste)
- **Pitfalls & fixes** ("If X → do Y")
- **What to change** (3–8 knobs likely to matter)

---

## Key PEFT concepts (quick reference)

### LoRA parameters
| Parameter | Description | Typical values |
|---|---|---|
| `r` | Rank of low-rank matrices | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor (effective = lora_alpha/r) | 16, 32 (often 2x of r) |
| `lora_dropout` | Dropout for LoRA layers | 0.05, 0.1 |
| `target_modules` | Which layers to apply LoRA | `["q_proj", "v_proj"]`, `"all-linear"` |
| `modules_to_save` | Layers to fully train (e.g., classifier head) | `["classifier"]`, `["lm_head"]` |
| `task_type` | Task for proper head handling | `CAUSAL_LM`, `SEQ_CLS`, `SEQ_2_SEQ_LM` |

### Memory comparison (typical 7B model)
| Method | VRAM (16GB GPU) | Trainable % |
|---|---|---|
| Full fine-tuning | 60+ GB (OOM) | 100% |
| LoRA (fp16) | ~14 GB | 0.1-0.5% |
| QLoRA (4-bit) | ~6-8 GB | 0.1-0.5% |

---

## Guardrails (non-negotiable)
- Do not invent PEFT parameters/behavior. Verify if uncertain.
- Do not recommend full fine-tuning when PEFT solves the memory constraint.
- Always specify `task_type` in LoraConfig for proper head handling.
- When using quantization, always use `prepare_model_for_kbit_training()`.
- Keep PEFT responsibilities separate from Transformers core unless integration is the blocker.
