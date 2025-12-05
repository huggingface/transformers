# DeepSeek V3.2 HF vs Reference – TODO Alignment

This document tracks the functional deltas between `src/transformers/models/deepseek_v32/modular_deepseek_v32.py`
and the reference implementation in `src/transformers/models/deepseek_v32/reference/model.py`. Each subsection
summarizes what must be fixed or reworked to achieve feature parity (excluding FP8 quantization, which we
intentionally skip for the initial bf16 port).

## 1. Tensor Parallelism & Sharded Layers
- Reference relies on `ParallelEmbedding`, `ColumnParallelLinear`, and `RowParallelLinear` with `world_size` /
  `rank` awareness and cross-rank reductions. HF now ships a conversion utility that reconstructs dense HF weights
  from the reference tensor-parallel shards.
- **Status:** Use `src/transformers/models/deepseek_v32/convert_deepseek_v32_reference_checkpoint.py` to merge
  per-rank checkpoints into a single `.safetensors` file that can be loaded with `DeepseekV32ForCausalLM`.
  Example:

  ```bash
  python -m transformers.models.deepseek_v32.convert_deepseek_v32_reference_checkpoint \
      --shard_paths rank0.safetensors rank1.safetensors ... \
      --config ./config_671B_v3.2.json \
      --output ./deepseek_v32_dense.safetensors \
      --dtype bfloat16
  ```

- **Next steps:** Long term we should still add true tensor-parallel layers, but reference checkpoints can now be
  converted and loaded into the dense HF implementation.

## 2. Dense vs MoE Layer Scheduling
- Reference toggles between dense and MoE via `n_dense_layers`, which is functionally identical to HF’s
  `first_k_dense_replace`. The HF decoder already switches to MoE once `layer_idx >= config.first_k_dense_replace`
  (see `DeepseekV32DecoderLayer`), so parity is preserved as long as the config value matches the reference
  checkpoint.
- HF’s default of `first_k_dense_replace=3` is taken from the released 671B reference config
  (`n_dense_layers: 3` in `config_671B_v3.2.json`). For smaller reference configs that default to `n_dense_layers=1`,
  users can simply set `first_k_dense_replace=1` when instantiating `DeepseekV32Config`.
- **Status:** No change required beyond supplying the correct value in the config used for conversion/training.

## 3. MoE Routing Semantics
- HF already mirrors the reference gating rules. `DeepseekV32Gate` applies the same sigmoid/softmax scoring,
  optional bias, group-limited selection (`n_group`/`topk_group` via the `noaux_tc` default), and `routed_scaling_factor`
  before normalizing the top-k weights (see `DeepseekV32Gate.forward`). `DeepseekV32MoE` also adds the shared experts
  exactly like the reference (`DeepseekV32MoE.forward`).
- The only missing piece is distributed expert ownership / `dist.all_reduce`, which is tracked separately in Issue 10.
- **Status:** No additional routing work required for the single-process dense HF model; gating parity is already in place.

## 4. Lightning Indexer & Sparse Masking
- The HF indexer already implements the reference data path:
  - It applies the same Hadamard transform (`hadamard_transform_activation`) before scoring and reuses the
    compressed Q states (`q_compressed`) like the reference indexer (`DeepseekV32Indexer.forward`).
  - `cache_position` provides the same information as the reference `start_pos`, and `_update_cache` writes the new
    keys into a persistent buffer so that `topk_indices` are always computed against the full prefix
    (`DeepseekV32Indexer._update_cache`).
  - The sparse mask matches the reference contract for both prefill and decode, using `[B, S, T]` or `[B, 1, T]`
    scatter masks that gate the attention weights after applying the standard causal mask
    (`DeepseekV32Attention.forward`).
- We intentionally run the cache in bf16 (the reference fp8 path depends on tilelang kernels) and skip the
  distributed broadcast because the dense HF port currently targets single-process inference; the cross-rank sync is
  already called out in Issue 10.
- **Status:** No further work required for the single-process bf16 path; only the multi-rank broadcast piece remains
  open under Issue 10.

## 5. Attention / MLA Decode Path
- Reference distinguishes MHA prefill vs MQA decode, reusing `kv_cache` and `pe_cache`, and contracting decode to
  a single value head. HF always runs full multi-head attention with `DynamicCache`, so decode latency and masking
  differ.
- **Action:** Implement the dual-path logic (prefill MHA vs decode MQA), reuse the reference cache layout, and
  hook it into the Transformers `Cache` interface without recomputing full multi-head matmuls during decode.

## 6. Rotary / YaRN Frequencies
- Reference precomputes `freqs_cis` once per model and slices by `start_pos`. HF recomputes frequencies every
  forward pass using `position_ids`, which can drift from the reference float math.
- **Action:** Precompute and cache the YaRN-adjusted `freqs_cis` buffer (respecting `max_seq_len`) and pass only
  `start_pos`/`seqlen` to the layers, matching reference indexing.

## 7. Causal Mask Contract
- Reference builds a simple `[seqlen, seqlen]` upper-triangular mask for prefill and relies on `mask=None` during
  decode. HF uses the generic `_update_causal_mask` that combines `attention_mask` and `cache_position`, which can
  yield different mask shapes/values.
- **Action:** Reproduce the reference masking rules (triangular prefill, `None` for decode, add sparse mask from
  the indexer) and ensure shapes match what the attention / indexer expect.

## 8. Output / Logit Behavior
- Reference `Transformer` returns logits only for the last token. HF LM head produces full sequence logits (with
  optional truncation), changing how reference checkpoints’ heads map to HF training code.
- **Action:** Decide on consistent semantics: either adapt the HF LM head to mimic the reference (last-token only)
  or document + implement conversion logic so checkpoints trained with the reference head can be used for causal
  LM training without ambiguity.

## 9. Residual & Norm Handling
- Reference RMSNorm fuses residual addition (`attn_norm(x, residual)`), storing weights in float32. HF performs
  separate pre/post normalizations with default dtype parameters, so intermediate values differ.
- **Action:** Port the residual-aware RMSNorm (or justify the deviation) and ensure weight/bias dtypes line up with
  the reference to minimize numerical drift.

## 10. Distributed Expert Aggregation
- Reference MoE all-reduces expert outputs when `world_size > 1`. HF implementation keeps everything local, so
  multinode inference/training will not match.
- **Action:** Add optional `dist` integration that mirrors the reference’s `all_reduce` semantics, even if the
  default run is single-process.

## 11. API Parity (start_pos vs cache_position)
- Reference APIs revolve around `start_pos` integers and internal cache buffers sized by `max_batch_size`. HF
  exposes `cache_position` tensors and resets indexer caches on new generations.
- **Action:** Provide adapters so generation flows can pass `start_pos` as in the reference and still interact with
  HF’s `Cache`. This includes resetting indexer caches only when intended and keeping cache lengths synchronized.

---

Addressing the above items will bring the bf16 HF implementation in line with the functional behavior of the
reference DeepSeek V3.2 model (sans fp8 kernels).

