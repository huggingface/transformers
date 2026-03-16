# _static_sample: Remaining neuron_sample Optimizations Investigation

**Date:** 2026-03-16
**Branch:** `static_shape_generation`
**Baseline:** 6 commits implementing `_static_sample` with +1.3% speedup on CUDA vs `_sample`
**Hardware:** NVIDIA A10G, Llama-3.1-8B, torch 2.10, SDPA attention

---

## Items Evaluated

### Item 1: CPU-side EOS tracking

**Status: IMPLEMENTED — no measurable CUDA improvement**

**What was done:**
- Moved `unfinished_sequences` to CPU (`device="cpu"`)
- Extract `eos_token_id_cpu` from `EosTokenCriteria` before the loop
- Each step: `next_tokens_cpu = next_tokens.cpu()` (async D2H), then `torch.isin(next_tokens_cpu, eos_token_id_cpu)` on CPU
- `unfinished_sequences.max() == 0` early exit is now a CPU op (no device sync)
- EOS masking uses `unfinished_sequences.to(device)` (small H2D copy) for device-side `next_tokens` masking

**Rationale:** The `max()` reduction on a device tensor forces a blocking D2H sync every decode step. Moving to CPU replaces this with an async `next_tokens.cpu()` transfer + CPU-only bookkeeping.

**Result on CUDA:** Still +1.3% total (no change). On CUDA with torch.compile, the `max()` sync overhead is already negligible — the inductor backend pipelines these operations efficiently. **On Neuron/XLA, this matters more (~40ms per sync).**

**Code location:** `src/transformers/generation/utils.py` lines ~2922-2931 (setup), ~2976-2983 (first token), ~3076-3083 (decode loop)

---

### Item 2: 4D causal mask built once

**Status: SKIPPED — not worth the complexity on CUDA**

**Investigation findings:**
- The 2D→4D mask conversion happens inside the compiled graph (`_optimize_model_for_decode` context) and is **fused by inductor** on CUDA
- `masking_utils.py:828` has an early exit when mask is already 4D — pre-building would skip the conversion
- **Format problem:** SDPA expects `bool` 4D masks (True=attend), eager expects `float` (0.0=attend, min_dtype=masked). A pre-built mask must match the attention backend
- **Sliding window:** `neuron_sample` falls back to 2D for models with sliding attention layers. We'd need the same fallback
- Previous attempt (earlier in the branch history) produced incorrect outputs due to mask value convention mismatch

**Why skipped:** The 2D→4D conversion cost is near-zero on CUDA (fused into compiled graph). The implementation complexity (format-dependent, sliding window fallback) isn't justified. **Worth revisiting for Neuron/XLA** where each new op in the compiled graph triggers separate compilation.

**Key files for future work:**
- `src/transformers/masking_utils.py:828` — 4D early exit
- `src/transformers/masking_utils.py:957-959` — `is_compileable` blocks mask skip
- `src/transformers/masking_utils.py:610-612` — eager float mask format
- `src/transformers/masking_utils.py:367` — SDPA boolean mask format

---

### Item 3: Left-padding position_offset

**Status: IMPLEMENTED — correctness fix**

**Bug:** `_static_sample` computed `position_ids = cache_position` for all batch elements. With left-padded batches (required for decoder-only batched generation), different batch elements have different padding amounts, so `cache_position != position_id`.

**Fix:** Compute `position_offset` once before the loop:
```python
position_offset = prefill_len - attention_mask[:, :prefill_len].sum(dim=-1, keepdim=True)
```
Then each step: `position_ids = cache_position - position_offset`

**Code location:** `src/transformers/generation/utils.py` lines ~2991-3004 (setup), ~3017-3019 (decode loop)

**Note:** Not yet verified with a batched left-padded test — the greedy sanity check (batch_size=1) passes. **Need a batch_size>1 test with variable-length prompts to confirm.**

---

### Item 4: Dynamic output_ids[:, :cur_len] slices

**Status: DEFERRED**

**Current behavior:** `logits_processor(output_ids[:, :cur_len], ...)` passes a growing slice each step. The shape changes per step.

**Why it was initially dismissed:** `logits_processor` and `stopping_criteria` run outside `_optimize_model_for_decode()` (eager mode), so dynamic shapes don't cause graph breaks or recompilation.

**Why it needs revisiting:**
- If Item 1 moves `output_ids` to CPU (for Neuron), the slice is CPU → fine for CPU-side logits processing
- On CUDA, passing the growing slice to `RepetitionPenaltyLogitsProcessor` triggers `scatter_()` with the slice as index tensor — dynamic shape but still eager
- `neuron_sample` passes the **full buffer** (including unfilled pad positions) — simpler but exposes pads to processors. Need to check if processors handle trailing pads correctly

---

## Benchmark Summary (CUDA, A10G, Llama-3.1-8B)

Medium prompt (~83 tokens), 256 new tokens, greedy decoding, SDPA, torch.compile max-autotune:

| Configuration | Avg (s) | Tok/s | Speedup |
|---|---|---|---|
| `_sample` + static cache + compile | 8.020 | 31.9 | 1.00x |
| `_static_sample` + static cache + compile (all items) | 7.916 | 32.3 | 1.01x (+1.3%) |

Peak memory: identical (15,376.4 MB). Sanity check: PASSED (identical greedy output).

**Conclusion:** On CUDA with torch.compile/inductor, the additional optimizations (Items 1-2) don't provide measurable improvement beyond the original 5 `torch.cat` replacements. The inductor backend already optimizes away most of the overhead. These optimizations are primarily valuable for **Neuron/XLA** where dynamic shapes trigger recompilation and device syncs have ~40ms overhead.

---

## Current State of the Code

Uncommitted changes on `static_shape_generation` (on top of the 6 existing commits):
- Item 1: CPU-side EOS tracking
- Item 3: Left-padding position_offset

These changes are **not yet committed**. The 6 existing commits are:
```
c5c76f4 Add _static_sample method for static-shape generation with StaticCache
829830b Bypass prepare_inputs_for_generation in _static_sample decode loop
c73b95b Remove redundant past_key_values reassignment in _static_sample
6579403 Replace .item() device-host sync with loop index in _static_sample
9b3ef45 Replace scatter_ with direct indexing for output buffer in _static_sample
fd28f3b Auto-dispatch to _static_sample when StaticCache is detected
```

## Next Steps

1. **Test Item 3** with batched left-padded inputs (batch_size=2, different prompt lengths)
2. **Decide on Item 1** — keep or revert? No CUDA benefit, but prepares the code for Neuron
3. **Revisit Item 4** — dynamic slices vs full buffer for logits_processor
4. **Revisit Item 2** — only if targeting Neuron/XLA
5. **Update issue #44742** with findings from `project_static_sample_incremental_fixes.md`
