# DeepSeek V3.2 HF vs Reference – Alignment Status

This document tracks the functional deltas between `src/transformers/models/deepseek_v32/modular_deepseek_v32.py`
and the reference implementation in `src/transformers/models/deepseek_v32/reference/model.py`. Each subsection
summarizes the current status and any remaining work (excluding FP8 quantization, which we intentionally skip
for the initial bf16 port).

## 1. Tensor Parallelism & Sharded Layers
- Reference relies on `ParallelEmbedding`, `ColumnParallelLinear`, and `RowParallelLinear` with `world_size` /
  `rank` awareness and cross-rank reductions. HF now ships a conversion utility that reconstructs dense HF weights
  from the reference tensor-parallel shards.
- **Status:** ✅ Use `src/transformers/models/deepseek_v32/convert_deepseek_v32_reference_checkpoint.py` to merge
  per-rank checkpoints into a single `.safetensors` file that can be loaded with `DeepseekV32ForCausalLM`.
  Example:

  ```bash
  python -m transformers.models.deepseek_v32.convert_deepseek_v32_reference_checkpoint \
      --shard_paths rank0.safetensors rank1.safetensors ... \
      --config ./config_671B_v3.2.json \
      --output ./deepseek_v32_dense.safetensors \
      --dtype bfloat16
  ```

- **Note:** For distributed training, use FSDP or DeepSpeed ZeRO-3 instead of native TP layers (see §10).

## 2. Dense vs MoE Layer Scheduling
- Reference toggles between dense and MoE via `n_dense_layers`, which is functionally identical to HF's
  `first_k_dense_replace`. The HF decoder already switches to MoE once `layer_idx >= config.first_k_dense_replace`
  (see `DeepseekV32DecoderLayer`), so parity is preserved as long as the config value matches the reference
  checkpoint.
- HF's default of `first_k_dense_replace=3` is taken from the released 671B reference config
  (`n_dense_layers: 3` in `config_671B_v3.2.json`). For smaller reference configs that default to `n_dense_layers=1`,
  users can simply set `first_k_dense_replace=1` when instantiating `DeepseekV32Config`.
- **Status:** ✅ Complete. No change required beyond supplying the correct value in the config.

## 3. MoE Routing Semantics
- HF already mirrors the reference gating rules. `DeepseekV32Gate` applies the same sigmoid/softmax scoring,
  optional bias, group-limited selection (`n_group`/`topk_group` via the `noaux_tc` default), and `routed_scaling_factor`
  before normalizing the top-k weights (see `DeepseekV32Gate.forward`). `DeepseekV32MoE` also adds the shared experts
  exactly like the reference (`DeepseekV32MoE.forward`).
- **Status:** ✅ Complete. Routing logic matches reference. For distributed training, see §10 (FSDP/ZeRO-3).

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
- We intentionally run the cache in bf16 (the reference fp8 path depends on tilelang kernels).
- **Status:** ✅ Complete for bf16 path. FP8 is intentionally not supported.

## 5. Attention / MLA Decode Path
- Reference distinguishes MHA prefill vs MQA decode, reusing `kv_cache` and `pe_cache`, and contracting decode to
  a single value head. HF always runs full multi-head attention with `DynamicCache`, so decode latency and masking
  differ.
- **Status:** ⚠️ Functional but not optimized. Decode works correctly but doesn't use the optimized MQA path.
- **Future optimization:** Implement dual-path logic (prefill MHA vs decode MQA) for better inference latency.

## 6. Rotary / YaRN Frequencies
- Reference precomputes `freqs_cis` once per model and slices by `start_pos`. HF recomputes frequencies every
  forward pass using `position_ids`, which can drift from the reference float math.
- **Status:** ⚠️ Functional with minor numerical differences. Results are correct but may have small floating-point
  drift compared to reference.
- **Future optimization:** Precompute and cache the YaRN-adjusted `freqs_cis` buffer for exact numerical match.

## 7. Causal Mask Contract
- Reference builds a simple `[seqlen, seqlen]` upper-triangular mask for prefill and relies on `mask=None` during
  decode. HF uses the generic `_update_causal_mask` that combines `attention_mask` and `cache_position`, which can
  yield different mask shapes/values.
- **Status:** ⚠️ Functional. HF masking works correctly but uses a different internal representation than reference.

## 8. Output / Logit Behavior
- Reference `Transformer` returns logits only for the last token. HF LM head produces full sequence logits (with
  optional truncation via `logits_to_keep`), changing how reference checkpoints' heads map to HF training code.
- **Status:** ✅ Complete. HF follows standard Transformers conventions. Use `logits_to_keep` parameter if you only
  need the last token's logits for memory efficiency.

## 9. Residual & Norm Handling
- Reference RMSNorm fuses residual addition (`attn_norm(x, residual)`), storing weights in float32. HF performs
  separate pre/post normalizations with default dtype parameters, so intermediate values differ.
- **Status:** ⚠️ Functional with minor numerical differences. Results are correct but may have small floating-point
  drift compared to reference due to unfused residual + norm operations.

## 10. Distributed Training (FSDP / DeepSpeed ZeRO-3)
- Reference MoE uses manual Expert Parallelism (EP) with explicit `all_reduce` for expert outputs.
- **Status:** ✅ Complete via FSDP/ZeRO-3. The HF implementation uses `nn.ModuleList` of `nn.Linear` layers for
  experts instead of 3D parameter tensors, enabling automatic parameter sharding by distributed training frameworks.

### Usage with FSDP:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import DeepseekV32ForCausalLM, DeepseekV32Expert

model = DeepseekV32ForCausalLM(config)
# Optionally wrap each expert separately for fine-grained sharding
model = FSDP(model, auto_wrap_policy=ModuleWrapPolicy({DeepseekV32Expert}))
```

### Usage with DeepSpeed ZeRO-3:
```python
import deepspeed
from transformers import DeepseekV32ForCausalLM

ds_config = {
    "zero_optimization": {"stage": 3},
    "bf16": {"enabled": True},
    ...
}
model = DeepseekV32ForCausalLM(config)
model, optimizer, _, _ = deepspeed.initialize(model=model, config=ds_config)
# ZeRO-3 will partition expert parameters automatically
```

### Key Design Decisions:
- **Removed manual EP logic**: Instead of hardcoded `ep_size`/`ep_rank` and manual `all_reduce`, we let FSDP/ZeRO-3
  handle parameter sharding and gradient synchronization automatically.
- **ModuleList-based experts**: Each expert is a separate `DeepseekV32Expert` module with standard `nn.Linear` layers,
  which FSDP can wrap and shard independently.
- **No explicit `.to(device)` calls**: Removed device placement code that could interfere with FSDP/ZeRO-3's
  automatic device management.

## 11. API Parity (start_pos vs cache_position)
- Reference APIs revolve around `start_pos` integers and internal cache buffers sized by `max_batch_size`. HF
  exposes `cache_position` tensors and resets indexer caches on new generations.
- **Status:** ⚠️ Functional. HF uses standard Transformers cache API (`cache_position`, `DynamicCache`). Works
  correctly but API differs from reference.

---

## Summary

| Feature | Status | Notes |
|---------|--------|-------|
| Checkpoint Conversion | ✅ | Convert TP shards to dense |
| Dense/MoE Scheduling | ✅ | `first_k_dense_replace` |
| MoE Routing | ✅ | Sigmoid scoring, group selection |
| Lightning Indexer | ✅ | Hadamard transform, sparse masking |
| Distributed Training | ✅ | FSDP/ZeRO-3 compatible |
| Training API | ✅ | Stage 1 (SFT) + Stage 2 (KL loss) |
| Decode Optimization | ⚠️ | Functional, not MQA-optimized |
| YaRN Frequencies | ⚠️ | Minor numerical drift |
| Causal Masking | ⚠️ | Different internal representation |
| Residual + Norm | ⚠️ | Unfused operations |
| Cache API | ⚠️ | HF conventions vs reference |
| FP8 Quantization | ❌ | Intentionally not supported |

The bf16 HF implementation is functionally complete for training and inference. Items marked ⚠️ work correctly
but may have minor numerical differences or performance gaps compared to the reference implementation.

