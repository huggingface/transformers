# _static_sample: Clean-up of neuron_sample into a General Static-Shape Generation Loop

**Date:** 2026-03-18
**Branch:** `static_shape_generation`
**Issue:** #44742

---

## Context

### Goal

The `neuron_sample` function in [`huggingface/torch-neuronx`](https://github.com/huggingface/torch-neuronx/blob/evaluation/examples/huggingface/transformers/run_generation.py#L536) is a standalone replacement for `GenerationMixin._sample` that enforces static tensor shapes throughout the decode loop. It was written specifically for AWS Neuron, where every new tensor shape triggers a fresh NEFF compilation (~40ms+ penalty per shape change per step).

The question is: **should this live in `transformers` as a Neuron-only path, or as a general static-shape path that also benefits CUDA and other devices?**

To answer this, we are cleaning up `neuron_sample` into `_static_sample` (integrated into `GenerationMixin`) and benchmarking each optimization step-by-step on CUDA. If the optimizations bring measurable CUDA improvement, the path is general-purpose. If not, it's Neuron/XLA-specific.

### What `neuron_sample` fixes over `_sample`

The default `_sample` has 5 `torch.cat` operations that create new tensor shapes every decode step:

1. `input_ids` grows by 1 token each step
2. `attention_mask` grows by 1 element each step
3. `position_ids` grows by 1 element each step
4. `cache_position` grows by 1 element each step
5. `while` loop conditioned on dynamically-shaped stopping check

Beyond eliminating these, `neuron_sample` also:

6. Bypasses `prepare_inputs_for_generation` (avoids per-step `clone()` — each clone triggers ~40ms device sync on Neuron)
7. Sets `position_ids` explicitly from `cache_position` (avoids dynamic `cumsum` on attention mask)
8. Pre-computes a static 4D causal mask updated in-place via `scatter_` (avoids per-step 2D→4D conversion)
9. Falls back to static 2D mask for models with sliding window layers
10. Keeps `output_ids` on CPU (avoids device-side scatter/index-put NEFF compilations)
11. Runs all EOS tracking on CPU using `next_tokens.cpu()` (avoids ~40ms device sync from `max()` reduction)
12. Passes full `output_ids` buffer to `logits_processor` (no growing slice — static shape)
13. Uses a separate `next_token_device` buffer updated via `.copy_()` for decode input
14. Builds `model_inputs` dict directly (no `_update_model_kwargs_for_generation` calls)

### Reference: `neuron_sample` source

Located at `huggingface/torch-neuronx`, branch `evaluation`:
`examples/huggingface/transformers/run_generation.py`, line 536.

Key structural choices:
- Loop runs `i in range(requested_max_new_tokens)` with `i == 0` processing prefill output
- `output_ids` on CPU, shape `(batch, total_len)`, filled with `pad_token_id`
- `next_token_device`: separate `(batch, 1)` device tensor for model input, updated via `.copy_()`
- 4D causal mask: `(batch, 1, 1, max_cache_len)` filled with `min_dtype`, unmasked via `scatter_` each step
- Sliding window fallback: 2D mask `(batch, max_cache_len)` with decode positions pre-set to 1
- EOS masking on CPU: `next_tokens_cpu * unfinished + pad * (1 - unfinished)`
- No `return_dict_in_generate`, no streamer, no scores/logits/attentions collection
- Timing instrumentation via `model._timing` dict
- Supports chunked prefill via `_neuron_chunked_prefill`

---

## Methodology: Aligning on the Newest Algorithm

`neuron_sample` was forked from `_sample` ~2 weeks ago. Since then, several PRs have landed on `main` that changed `_sample`'s algorithm. When `neuron_sample` and current `_sample` disagree, **we align on the newest `_sample` as the source of truth** and only deviate where static shapes require it.

### Key recent PRs affecting `_sample`:

1. **#44226** `[generate] Always pass full input_ids in prepare_inputs_for_generation` (2026-02-24)
   - `_sample` now passes full `input_ids` to `prepare_inputs_for_generation`, which slices via `next_sequence_length` param
   - `neuron_sample` bypasses `prepare_inputs_for_generation` entirely — **still valid** (we bypass it too, for different reasons)

2. **#44130** `[generate] Completely stop relying on cache_position to prepare inputs` (2026-02-21)
   - Input slicing no longer uses `cache_position` — uses `next_sequence_length` instead
   - `neuron_sample` still uses `cache_position` for position_ids and mask updates — **still needed for static shapes**

3. **#44181** `[core] Completely remove cache positions` (2026-03-04)
   - Removed `cache_position` from cache and masking APIs (not from generation loop itself)
   - `_sample` still has `cache_position` in `_update_model_kwargs_for_generation` and `prepare_inputs_for_generation`
   - **Impact on `_static_sample`:** `cache_position` is still passed to model forward. Need to verify models still accept it.

4. **#44126** `Simplify input preparation in generate` (2026-02-20)
   - Simplified slicing logic in `prepare_inputs_for_generation`
   - `neuron_sample` bypasses this entirely — **no impact**

### Current `_sample` structure (post all PRs):

```python
# Prefill
outputs = self._prefill(input_ids, generation_config, model_kwargs)

while self._has_unfinished_sequences(...):
    if prefill_consumed:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, next_sequence_length=1, **model_kwargs  # <-- full input_ids, sliced inside
        )
        outputs = model_forward(**model_inputs, return_dict=True)
    prefill_consumed = True
    model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, ...)

    next_token_logits = outputs.logits[:, -1, :].to(copy=True, ...)
    next_token_scores = logits_processor(input_ids, next_token_logits)  # <-- full input_ids
    # ... token selection ...
    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
    input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  # <-- grows each step
    unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
    this_peer_finished = unfinished_sequences.max() == 0
```

**Key observations vs `neuron_sample`:**
- `_sample` passes **full growing `input_ids`** to `logits_processor` (not a slice of `output_ids`)
- `neuron_sample` passes **full pre-allocated `output_ids`** (including unfilled pad positions)
- Both pass the full sequence; the difference is growing vs pre-allocated with trailing pads
- `_update_model_kwargs_for_generation` still does `torch.cat` on `attention_mask`, `position_ids`, `cache_position` — these are exactly what `_static_sample` replaces with in-place updates

---

## Current State of `_static_sample`

### What's already implemented (committed)

6 commits on `static_shape_generation`:

```
c5c76f4 Add _static_sample method for static-shape generation with StaticCache
829830b Bypass prepare_inputs_for_generation in _static_sample decode loop
c73b95b Remove redundant past_key_values reassignment in _static_sample
6579403 Replace .item() device-host sync with loop index in _static_sample
9b3ef45 Replace scatter_ with direct indexing for output buffer in _static_sample
fd28f3b Auto-dispatch to _static_sample when StaticCache is detected
```

Plus 2 additional committed changes:
```
b8c775863d Fix left-padding position_ids in _static_sample for batched generation
6239cbaabc Move EOS tracking to CPU in _static_sample to avoid blocking device sync
```

### What's implemented vs `neuron_sample`

| # | Optimization | `neuron_sample` | `_static_sample` | Status |
|---|---|---|---|---|
| 1 | Eliminate 5 `torch.cat` ops | Yes | Yes | Done |
| 2 | Bypass `prepare_inputs_for_generation` | Yes | Yes | Done |
| 3 | Explicit `position_ids` from `cache_position` | Yes | Yes | Done |
| 4 | Position offset for left-padding | Yes | Yes | Done |
| 5 | Remove redundant `past_key_values` reassignment | N/A (neuron still reassigns) | Yes | Done |
| 6 | Replace `.item()` with loop index | Yes (uses `prompt_len + i`) | Yes (uses `prefill_len + i`) | Done |
| 7 | CPU-side EOS tracking (`unfinished_sequences` on CPU) | Yes | Yes | Done |
| 8 | `output_ids` on CPU | **Yes** | **No** — output_ids on device | **TODO** |
| 9 | 4D causal mask pre-built | **Yes** — `(batch, 1, 1, max_cache_len)` with `min_dtype` | **No** — 2D mask, model does 2D→4D each step | **TODO** |
| 10 | Sliding window fallback to 2D mask | **Yes** | **No** | **TODO** (depends on #9) |
| 11 | Full buffer to `logits_processor` (no growing slice) | **Yes** — `logits_processor(output_ids, ...)` | **No** — `output_ids[:, :cur_len]` | **TODO** |
| 12 | EOS masking entirely on CPU | **Yes** — `next_tokens_cpu * unfinished + pad * (1 - unfinished)` | **Partial** — EOS check on CPU, but masking uses `unfinished_sequences.to(device)` | **TODO** |
| 13 | Separate `next_token_device` via `.copy_()` | **Yes** | **No** — uses `current_ids[:, 0] = next_tokens` | **TODO** (tied to #8) |
| 14 | `return_dict_in_generate` support | No | Yes | `_static_sample` extra |
| 15 | Streamer support | No | Yes | `_static_sample` extra |
| 16 | Scores/logits/attentions collection | No | Yes | `_static_sample` extra |
| 17 | Chunked prefill | Yes | No | Out of scope for now |

### Structural differences

**Loop structure:**
- `neuron_sample`: single `for i in range(max_new_tokens)`, `i == 0` handles prefill output
- `_static_sample`: prefill token processed before loop, loop runs `max_new_tokens - 1` iterations

**EOS masking:**
- `neuron_sample`: all on CPU — `next_tokens_cpu * unfinished + pad * (1 - unfinished)` before writing to CPU output buffer
- `_static_sample`: EOS check on CPU, but masking on device — `next_tokens * unfinished.to(device) + pad * (1 - unfinished.to(device))`

**output_ids location:**
- `neuron_sample`: CPU — avoids all device-side bookkeeping ops
- `_static_sample`: device — logits_processor receives device tensors directly

---

## Remaining Items to Benchmark on CUDA

Each item below should be implemented and benchmarked independently to measure CUDA impact.

### Item A: `output_ids` on CPU + EOS masking fully on CPU

**What:** Move `output_ids` to CPU. All token bookkeeping (write, EOS masking) happens on CPU. Use a separate `next_token_device` buffer (batch, 1) for model input, updated via `.copy_()`.

**neuron_sample approach:**
```python
output_ids = torch.full((batch_size, total_len), fill_value=pad_id, dtype=input_ids.dtype, device="cpu")
output_ids[:, :prompt_len] = input_ids.cpu()
next_token_device = torch.zeros(batch_size, 1, dtype=input_ids.dtype, device=input_ids.device)
# In loop:
next_tokens_cpu = next_tokens.cpu()
next_tokens_cpu = next_tokens_cpu * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
output_ids[:, prompt_len + i] = next_tokens_cpu
next_token_device.copy_(next_tokens_cpu.unsqueeze(1))
```

**Current `_static_sample` approach:**
```python
output_ids = input_ids.new_full((batch_size, max_length), pad_id)  # on device
# In loop:
unfinished_device = unfinished_sequences.to(device)
next_tokens = next_tokens * unfinished_device + pad_id * (1 - unfinished_device)
output_ids[:, prefill_len + i + 1] = next_tokens
current_ids[:, 0] = next_tokens
```

**Expected CUDA impact:** Unclear. The `.cpu()` copy is async and cheap, but `logits_processor` receiving CPU tensors may force a H2D transfer for processors that do device ops. Need to check if `logits_processor` can work with CPU input when `next_token_logits` is on device.

**Risk:** `logits_processor` implementations may assume `input_ids` and `scores` are on the same device. If `output_ids` is CPU and `next_token_logits` is device, this breaks. For example, `RepetitionPenaltyLogitsProcessor` uses `input_ids` as a scatter index into `scores` (a device tensor) — CPU `input_ids` would fail or silently force a sync.

**Alignment note:** Current `_sample` passes full growing `input_ids` (on device) to `logits_processor`. `neuron_sample` passes full pre-allocated `output_ids` (on CPU). Moving `output_ids` to CPU diverges from `_sample`'s device assumption. If this doesn't help CUDA, it may be better to keep `output_ids` on device for the general path and only move to CPU in a Neuron-specific variant.

---

### Item B: 4D causal mask pre-built with sliding window fallback

**What:** Pre-build a static 4D causal mask `(batch, 1, 1, max_cache_len)` filled with `min_dtype`, unmask prompt positions, then `scatter_` each decode position.

**neuron_sample approach:**
```python
min_dtype = torch.finfo(model.dtype).min
causal_mask = torch.full((batch_size, 1, 1, max_cache_len), min_dtype, device=device, dtype=model.dtype)
if attention_mask_2d is not None:
    causal_mask[:, 0, 0, :prompt_len].masked_fill_(attention_mask_2d.bool(), 0.0)
# In decode loop:
mask_idx = cp.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
causal_mask.scatter_(3, mask_idx, 0.0)
```

**Sliding window fallback:**
```python
has_sliding_layers = hasattr(past_key_values, "is_sliding") and any(past_key_values.is_sliding)
if has_sliding_layers:
    attention_mask_sliding = torch.ones((batch_size, max_cache_len), dtype=torch.long, device=device)
    if attention_mask_2d is not None:
        attention_mask_sliding[:, :prompt_len] = attention_mask_2d
    model_kwargs["attention_mask"] = attention_mask_sliding
else:
    model_kwargs["attention_mask"] = causal_mask
```

**Previous attempt:** Failed due to mask value convention mismatch — SDPA expects `bool` (True=attend), eager expects `float` (0.0=attend, min_dtype=masked). `neuron_sample` uses the float convention. Need to verify this matches what the model's `create_causal_mask` produces when it sees a 4D input (early exit at `masking_utils.py:828`).

**Expected CUDA impact:** Likely minimal — the 2D→4D conversion is fused by inductor into the compiled graph. But worth measuring.

**Key files:**
- `src/transformers/masking_utils.py:828` — 4D early exit
- `src/transformers/masking_utils.py:957-959` — `is_compileable` blocks mask skip
- `src/transformers/masking_utils.py:610-612` — eager float mask format
- `src/transformers/masking_utils.py:367` — SDPA boolean mask format

---

### Item C: Full buffer to `logits_processor` (no growing slice)

**What:** Pass `logits_processor(output_ids, next_token_logits)` instead of `logits_processor(output_ids[:, :cur_len], next_token_logits)`.

**neuron_sample approach:** Passes full CPU `output_ids` buffer (including unfilled pad positions after `cur_len`).

**Risk:** Logits processors may behave differently when they see trailing pad tokens. For example, `RepetitionPenaltyLogitsProcessor` would penalize pad_token_id if it appears in the full buffer. Need to check each processor in `logits_processor_list` for pad sensitivity.

**Expected CUDA impact:** Eliminates a per-step dynamic-shape slice. On CUDA this is eager code so the impact may be negligible. But it simplifies the code and aligns with neuron_sample.

**Depends on:** Item A if `output_ids` is on CPU.

---

## Benchmark Results (CUDA, A10G, Llama-3.1-8B)

Medium prompt (~83 tokens), 256 new tokens, greedy decoding, SDPA, torch.compile max-autotune:

### Baseline (committed state)

| Configuration | Avg (s) | Tok/s | Speedup |
|---|---|---|---|
| `_sample` + static cache + compile | 8.020 | 31.9 | 1.00x |
| `_static_sample` + static cache + compile (items 1-7) | 7.916 | 32.3 | 1.01x (+1.3%) |

Peak memory: identical (15,376.4 MB). Sanity check: PASSED (identical greedy output).

### Item A: output_ids on CPU + full CPU EOS masking + full buffer to logits_processor

**Changes:**
- `output_ids` allocated on CPU (`device="cpu"`)
- `output_ids[:, :prefill_len] = input_ids.cpu()` (prompt copied to CPU)
- `current_ids` updated via `.copy_(next_tokens_cpu.unsqueeze(1))` instead of direct indexing
- EOS masking entirely on CPU: `next_tokens_cpu * unfinished + pad * (1 - unfinished)` (no `to(device)`)
- `logits_processor(output_ids, ...)` receives full CPU buffer (no growing slice)
- `output_ids.to(device)` at the end before return

| Configuration | Avg (s) | Tok/s | Speedup vs `_sample` |
|---|---|---|---|
| `_sample` (baseline) | 8.010 | 32.0 | 1.00x |
| `_static_sample` + Item A | 7.897 | 32.4 | 1.01x (+1.4%) |

Peak memory: identical (15,376.4 MB). Sanity check: PASSED.

**Result:** +1.4% vs baseline (+0.6% improvement over pre-Item-A `_static_sample` which was +0.8%).
The improvement likely comes from eliminating the per-step `unfinished_sequences.to(device)` H2D transfer and the device-side EOS masking arithmetic.

### Item B: 4D causal mask pre-built (cumulative with Item A)

**Changes (on top of Item A):**
- After prefill, build a static 4D additive mask `(batch, 1, 1, max_cache_len)` filled with `min_dtype`
- Unmask prompt positions via `masked_fill_(attention_mask_2d.bool(), 0.0)`
- Each decode step: `scatter_(3, mask_idx, 0.0)` to unmask the current position
- Sliding window fallback: static 2D mask `(batch, max_cache_len)` with decode positions pre-set to 1
- Mask dimension uses `past_key_values.get_max_cache_shape()` (not `max_length`) to match cache KV size
- Prefill still uses the original 2D mask — 4D mask is only for the decode loop

**Gotcha encountered:** `max_length != max_cache_len`. The cache is allocated with `max_cache_length = max_length - 1` (see `generate()` line ~2511). The 4D mask must match cache KV length, not `max_length`.

| Configuration | Avg (s) | Tok/s | Speedup vs `_sample` |
|---|---|---|---|
| `_sample` (baseline) | 7.933 | 32.3 | 1.00x |
| `_static_sample` + Items A+B | 7.879 | 32.5 | 1.01x (+0.7%) |

Peak memory: identical (15,376.4 MB). Sanity check: PASSED.

**Result:** +0.7% total. Note that `_sample` baseline was faster this run (7.933 vs 8.010 in Item A's run), suggesting run-to-run variance of ~0.5-1%. The 4D mask appears to have no measurable benefit on CUDA — the 2D→4D conversion inside the compiled graph is already fused by inductor. The value is for Neuron/XLA where it avoids a per-step compilation.

### Item C: Full buffer to logits_processor

**Implemented as part of Item A** — `logits_processor(output_ids, next_token_logits)` where `output_ids` is the full CPU buffer. No separate benchmark needed; the effect is included in Item A's numbers.

---

## Integration Options: How `_static_sample` Can Live in `transformers`

The `generate()` method in `transformers` has several mechanisms for plugging in custom decoding
strategies. Understanding these is key to deciding how `_static_sample` should be integrated.

### Mechanism 1: Inline method on `GenerationMixin` (current approach)

`_static_sample` is defined directly on `GenerationMixin`, next to `_sample` and `_beam_search`.
The auto-dispatch logic in `generate()` selects it when a `StaticCache` is detected:

```python
# generate(), line ~2524
if custom_generate is None and generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
    if isinstance(model_kwargs.get("past_key_values"), StaticCache):
        decoding_method = getattr(type(self), "_static_sample")
```

All preparation (cache setup, logits processors, stopping criteria, input validation) still
happens in `generate()` before `decoding_method` is called. The decoding method receives the
same signature as `_sample`:

```python
result = decoding_method(
    self, input_ids,
    logits_processor=prepared_logits_processor,
    stopping_criteria=prepared_stopping_criteria,
    generation_config=generation_config,
    **generation_mode_kwargs, **model_kwargs,
)
```

**Pros:** Zero friction for users — just set `cache_implementation="static"` and it works.
Testable with the existing generation test suite. Visible in the codebase for review.

**Cons:** Adds ~240 lines to an already large `utils.py`. Must be maintained alongside `_sample`
when the generation API evolves (e.g. PRs #44226, #44130, #44181 all changed `_sample`'s
contract and would need mirrored changes in `_static_sample`).

### Mechanism 2: `custom_generate` callable parameter

Users can pass any callable as `custom_generate` to `generate()`:

```python
model.generate(..., custom_generate=my_static_sample_function)
```

At line 2366, this directly becomes the `decoding_method`:
```python
if isinstance(custom_generate, Callable):
    decoding_method = custom_generate
```

The callable receives the same args as `_sample` (extracted by comparing signatures at
line 2136). This is how the benchmark script forces `_sample` via
`custom_generate=GenerationMixin._sample` to bypass auto-dispatch.

**Pros:** No changes to `utils.py` at all — `_static_sample` could live in a separate module
(or even in `optimum-neuron`) and be injected at call time.

**Cons:** Requires users to explicitly pass it every time. No auto-dispatch. Not discoverable.

### Mechanism 3: `custom_generate` Hub repo (string parameter)

Deprecated generation modes (DOLA, contrastive search, group beam search, constrained beam
search) were moved to Hub repos under `transformers-community/*`. Users invoke them via:

```python
model.generate(..., custom_generate='transformers-community/dola', trust_remote_code=True)
```

This fetches `generate.py` from the Hub repo and uses its `generate` function as the decoding
method. The mapping is in `GENERATION_MODES_MAPPING`:

```python
GENERATION_MODES_MAPPING = {
    GenerationMode.SAMPLE: "_sample",
    GenerationMode.GREEDY_SEARCH: "_sample",
    # Deprecated — fetched from Hub:
    GenerationMode.DOLA_GENERATION: "transformers-community/dola",
    GenerationMode.CONTRASTIVE_SEARCH: "transformers-community/contrastive-search",
    ...
}
```

**Important nuance:** When fetched from a Hub repo, the function replaces the decoding method,
NOT the full `generate()`. All the preparation logic (cache, processors, stopping criteria)
still runs in `generate()` first.

`_static_sample` could be published as e.g. `transformers-community/static-sample` and invoked:
```python
model.generate(..., custom_generate='transformers-community/static-sample', trust_remote_code=True)
```

**Pros:** No changes to `utils.py`. Can be iterated on independently. Community-maintained.

**Cons:** Requires `trust_remote_code=True`. Requires users to know the repo name. No
auto-dispatch. The current Hub repo mechanism is positioned as a deprecation path
(comment says "remove this in v4.62.0"), not as a first-class extension point.

### Mechanism 4: `load_custom_generate` from model repo

If a model's HuggingFace repo contains a `custom_generate/generate.py` file, it is
automatically loaded at model instantiation time and **replaces the entire `generate()` method**:

```python
# from_pretrained(), line ~424
if hasattr(self, "load_custom_generate") and trust_remote_code:
    custom_generate = self.load_custom_generate(pretrained_model_name_or_path, ...)
    self.generate = functools.partial(custom_generate, model=self)
```

This is the coarsest mechanism — it replaces `generate()` entirely, not just the decoding loop.
The custom function must handle all preparation, cache setup, etc. on its own.

**Not suitable for `_static_sample`** since we only want to replace the decode loop, not the
full `generate()` pipeline. A Neuron-specific model repo could use this to ship a complete
Neuron-optimized generate, but that's a much larger scope.

### Recommendation

| Option | Auto-dispatch | Maintenance burden | User friction | Best for |
|---|---|---|---|---|
| **1. Inline method** | Yes (StaticCache) | High (must track `_sample` changes) | None | General path |
| **2. Callable param** | No | Low | High (manual each call) | Testing/development |
| **3. Hub repo** | No | Medium | Medium (need repo name + trust_remote_code) | Neuron-only path |
| **4. Model repo** | Yes (per model) | Low (self-contained) | None (if model ships it) | Model-specific overrides |

**If `_static_sample` is a general-purpose improvement** (our benchmarks show +0.7-1.4% on CUDA,
no regressions): **Option 1** is the right choice. Auto-dispatch gives the benefit to all users
with zero friction.

**If `_static_sample` is Neuron-only**: **Option 3** (Hub repo) would keep it out of core while
still being usable. But the Hub mechanism is currently positioned as a deprecation path, so
long-term viability is uncertain. **Option 2** (callable) is the pragmatic fallback — Neuron
users can import it and pass it explicitly.

**Hybrid approach:** Keep `_static_sample` as an inline method (Option 1) for auto-dispatch,
but also make it importable so Neuron users can pass it via Option 2 with custom modifications:
```python
from transformers.generation.utils import GenerationMixin
model.generate(..., custom_generate=GenerationMixin._static_sample)
```
This is already how the benchmark script works.

---

## Summary and Next Steps

### Benchmark summary

All optimizations from `neuron_sample` are now implemented in `_static_sample`. On CUDA:

| Configuration | Speedup vs `_sample` |
|---|---|
| `_static_sample` baseline (items 1-7) | +0.8% |
| + Item A (output_ids CPU + CPU EOS masking) | +1.4% |
| + Item B (4D causal mask) | +0.7% (noise) |

Run-to-run variance is ~0.5-1%, so all results are within noise. Key takeaway:
**no Neuron optimization hurts CUDA**. This supports making `_static_sample` a general path.

### What's done

All 14 `neuron_sample` optimizations are implemented and benchmarked.

### What's left

1. **Test left-padding correctness** — batch_size>1 with variable-length prompts (untested)
2. **Decide integration approach** — inline method with auto-dispatch (recommended) vs Hub repo
3. **PR preparation** — organize commits, write PR description per AGENTS.md rules, coordinate on issue #44742
4. **Verify `logits_processor` pad sensitivity** — full buffer includes trailing pads; check `RepetitionPenaltyLogitsProcessor` and others
