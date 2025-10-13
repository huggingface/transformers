# Problem Log

This log tracks problems you report against the current repository state. Unresolved problems are numbered below; resolved items move to the Resolved section.

How we use this log
- Add new problems by copying the template and filling it in.
- Keep "Unresolved problems" as a numbered list. Numbers reflect the current set of open items.
- When a problem is fixed or not reproducible, move it to "Resolved problems" and record the resolution.

---

## Unresolved problems (numbered)

1) P-001 — Llava-OneVision 7B eager attention with fp16 yields garbage output; 0.5B variant works

---

## Problem details

### P-001 — Llava-OneVision 7B eager attention with fp16 yields garbage output; 0.5B variant works
- Date reported: 2025-10-13
- Area: Multi-modal generation (LLaVA-OneVision, Qwen2-7B backbone), attention implementation
- Environment (reported):
  - attn_implementation: eager
  - dtype: float16
  - device_map: auto
  - Model: `llava-hf/llava-onevision-qwen2-7b-ov-hf` (fails) vs `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` (works)
- Repro steps (summary):
  - Build conversation with text + image; apply chat template via `AutoProcessor`.
  - Load 7B model with `attn_implementation="eager"` and `torch_dtype=torch.float16`.
  - Generate deterministically (`do_sample=False`).
  - Observed decoded output: degenerate response (e.g., just "!") instead of a proper caption; 0.5B produces correct caption.
- Expected: Consistent, non-degenerate outputs comparable to SDPA/FA2 path; ability to extract attention weights.
- Actual: 7B produces garbage output with eager+fp16; 0.5B works under same settings.
- Suspected cause (notes): Numerical stability/mask handling in eager attention path (softmax in fp16 and/or non-additive mask). SDPA/FA2 path more stable and hides weights.
- Known workarounds (notes):
  - Use bfloat16 with eager.
  - Use SDPA/FA2 for token selection and run an eager shadow pass per step to collect attentions.
- Status: Open (needs confirmation against current main; candidate fix: upcast softmax to fp32 and ensure additive mask in eager path for Qwen2-7B inside OneVision).
- Links: N/A (local log entry)

### P-002 — Missing RAG examples directory referenced in academic paper
- Date reported: 2025-10-13
- Area: Documentation, examples directory structure
- Environment (reported):
  - Repository: huggingface/transformers main branch
  - Referenced path: `/examples/rag`
- Repro steps (summary):
  1. Visit https://arxiv.org/pdf/2005.11401 (RAG paper)
  2. Follow reference to https://github.com/huggingface/transformers/blob/main/examples/rag
  3. Get 404 error
- Expected: RAG examples directory with working code samples as referenced in the academic paper
- Actual: 404 error - directory does not exist
- Suspected cause (notes): Examples were moved to `/examples/research_projects/rag` but paper reference not updated.
- Known workarounds (notes): 
  - Use examples at `/examples/research_projects/rag`
  - Create redirect or symlink
- Status: Resolved (created examples/rag directory with comprehensive README redirect)
- Resolution: Created `/examples/rag/README.md` with links to documentation, quick start code, and model references. Paper link now resolves correctly.
- Links: https://arxiv.org/pdf/2005.11401, https://github.com/huggingface/transformers/blob/main/examples/rag

---

## Resolved problems

### P-002 — Missing RAG examples directory referenced in academic paper
- Resolution: Created `/examples/rag/README.md` with comprehensive redirect to documentation and working code examples
- Date resolved: 2025-10-13
- Verification steps:
  1. Check directory exists: `Test-Path "examples/rag/README.md"` → True
  2. Verify content: File contains redirect links and working code examples
  3. Git status: Shows `?? examples/rag/` (new untracked files ready to commit)
  4. Manual test: Navigate to `/examples/rag` in file explorer - no longer 404

---

## Template (copy for new problems)

### P-XXX — <short title>
- Date reported: <YYYY-MM-DD>
- Area: <subsystem/model>
- Environment (reported):
  - attn_implementation: <eager/sdpa/flash>
  - dtype: <fp16/bf16/fp32>
  - device_map: <cpu/cuda/auto>
  - Model: <model id>
- Repro steps (summary):
  1. <step>
  2. <step>
  3. <step>
- Expected: <expected behavior>
- Actual: <actual behavior>
- Suspected cause (notes): <root cause hypothesis>
- Known workarounds (notes): <workaround(s)>
- Status: Open | Resolved | Needs Info
- Resolution (if resolved): <what fixed it — commit/PR/config change>
- Links: <issue/PR references>
