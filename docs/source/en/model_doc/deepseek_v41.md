<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-09-11.*

# DeepSeek-V4.1

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai) is the next MoE language model from DeepSeek. V4.1 keeps
V4's shared-K=V latent attention, Manifold-Constrained Hyper-Connections (mHC) and grouped low-rank output
projection, and replaces the fixed three-type attention schedule with a **ratio-scheduled** compressed-attention
design (CSA2), adds a **two-level indexer** (a candidate pre-filter constrains every later indexer), and introduces
the **engram** — deterministic n-gram hash lookups injected into the residual stream at a few layers.

This implementation covers the text backbone of `DeepSeek-V4.1-Flash` (and its `-Base` sibling):
[`DeepseekV41ForCausalLM`]. The MTP draft head (DSpark) and the vision tower ship in the released checkpoint but are
out of scope here: their keys are ignored on load and the corresponding config sections are accepted but unused.

## Architecture

### Ratio-scheduled compressed attention (CSA2)

`config.compress_ratios[i]` assigns every layer a compression ratio:

* `0` — pure sliding-window attention, no long-range branch.
* `r > 1` — the layer pools every `r` consecutive tokens into one shared KV latent with a learned softmax gate
  (`DeepseekV41Compressor`) and attends over `index_topk` of the pooled entries per query, on top of the usual
  sliding window.
* `1` — same machinery with one token per group: a plain compressed KV without pooling.

Layers sharing a ratio share **one** compressed KV and one indexer; the first of them (the *KV source*,
`config.kv_source_layer_ids`) owns the compressor that produces the latents, and the first index-capable one
(`config.index_source_layer_ids`) publishes the per-query top-k. Later layers of the same ratio reuse that state
("Reuse" mode); a later *index source* re-scores the same keys with its own weights ("Reindex" mode).

Every layer of the schedule also keeps the V4 backbone: shared K=V multi-query attention, partial RoPE with the
inverse rotation applied to the output's rope slice, per-head learnable attention sinks, and the grouped low-rank
output projection (`o_groups` × `o_lora_rank`).

### Two-level indexer

`DeepseekV41Indexer` scores each query against one key per compressed group (derived from the pre-RoPE compressor
latent) and keeps the top `index_topk` groups. When `config.candidate_source_layer_id` is set, the candidate source
layer first runs a coarser pass — `select_candidate_blocks` keeps the `candidate_topk_blocks` best-scoring blocks
of `candidate_block_size` compressed positions — and every later index source may only pick inside those blocks.

A group becomes visible to a query once the query has passed the group's last token (`(position + 1) // ratio`
visible groups in absolute positions), so chunked prefill, decode and one-shot prefill see exactly the same groups.

### Manifold-Constrained Hyper-Connections (mHC)

The residual stream is carried as `hc_mult` parallel copies. Each attention / FFN site mixes the streams with a
row-normalized (Sinkhorn-projected) mixing matrix; the site's pre-mix is computed one site ahead, so the pipeline
crosses layer boundaries exactly as in the released implementation. The raw layer parameters (`hc_attn_fn`,
`hc_attn_base`, `hc_attn_scale`, `hc_ffn_*`) and the attention sinks stay in `float32` — matching the released
checkpoint's dtypes (`_keep_in_fp32_modules_strict`).

### Engram

At `config.engram_layer_ids`, `DeepseekV41Engram` adds n-gram hash-table lookups into the residual stream: each
position is hashed with its `engram_max_ngram_size - 1` predecessor tokens (multiplied by per-layer random
multipliers, folded into prime-sized buckets), and each of the `engram_n_heads` heads reads one embedding row per
n-gram size. The hash is a pure function of `(tokenizer, config)` — no learned table is involved in the id mapping.

The state therefore needs the **tokenizer** to build its compressed token map. Bind one explicitly, or rely on the
auto-binding from `config._name_or_path` on the first forward:

```python
model.model.bind_tokenizer(tokenizer)  # explicit
# or: model = DeepseekV41ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
#    (the tokenizer is fetched from the same repo id on first use)
```

Tokens masked out of n-grams (image spans) are hashed as a DEAD sentinel; look-back stops at them, so an n-gram
never spans one.

## Quantization

### QAT fake-quant in the forward pass

The model is QAT-trained with activation quantization **baked into the forward pass** — it runs in every engine's
reference path and is applied here too, regardless of weight dtype:

| Tensor | Format | Scale |
| --- | --- | --- |
| sliding-window K=V (post-RoPE, pre-cache) | FP8 e4m3, 32-channel blocks | ue8m0 (power of two) |
| compressed KV latent (post-RoPE, pre-cache) | FP4 e2m1, 16-channel blocks | e4m3 |
| indexer keys / queries (post-RoPE) | FP4 e2m1, 32-channel blocks | ue8m0 |

This is model semantics, not a load-time effect: it is output-visible even with unquantized weights (FP4 rounding
moves indexer scores and therefore top-k selection). Tensors whose trailing dimension is not divisible by the block
size (tiny test configs) skip the quantization.

### Loading the released checkpoint

The released `DeepSeek-V4.1-Flash` checkpoint ships mixed-precision weights: attention projections, the shared
experts, `engram.wkv` and the engram tables in FP8 (e4m3, 32×32 blocks, ue8m0 scales), **routed experts packed as
FP4** (e2m1 nibbles in an int8 container, per-row 16-channel ue8m0 scales), the compressor / indexer projections,
embeddings and head in bf16, and the mHC / sink / gate-bias parameters in fp32. `from_pretrained` accepts this
layout as-is on CPU (weights are dequantized to the requested `dtype`) and on GPU (kept quantized when supported):
the fp8 quantizer's `.scale` → `weight_scale_inv` mapping and its FP4-aware dequantize op handle every piece,
including the packed experts and the per-row-scaled engram tables (dequantized into the embedding at load, so the
lookup stays a plain row gather).

`DeepseekV41ForCausalLM` takes the composite [`DeepseekV41Config`] (`config_class`) because the released
config.json nests the text backbone under `text_config` and puts `quantization_config` at the top level — pointing
the class at the bare text config would silently drop the quantization config and corrupt the fp8 tensors on load.
The text config is unwrapped in `__init__`; instantiating with a [`DeepseekV41TextConfig`] directly also works.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash", dtype="bfloat16")
inputs = tok("Hello, my dog is cute", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))
```

> [!NOTE]
> The model runs **eager attention only** (`_supports_flash_attn = _supports_sdpa = _supports_flex_attn = False`):
> `head_dim` exceeds what FlashAttention covers, the per-head attention sink needs a custom denominator, and the
> K=V cache is concatenated with the compressed branch after masking. SDPA / Flash kernels are future work.
> `StaticCache` / `QuantizedCache` are not supported either — the compressor's group state is dynamic
> (`DynamicCache`, the default, builds the right layers automatically).

## Implementation notes

* **Checkpoint-native naming.** The released checkpoint uses the reference implementation's names (`wq_a`, `ffn.experts.E.w1`, `engram.embed.weight`, `hc_attn_fn` on the layer, top-level `embed` / `norm` / `head`). The modules keep
  those names verbatim; `from_pretrained` only applies the four top-level renames registered centrally in
  `conversion_mapping.py` (`layers.*` → `model.layers.*`, `embed.weight` → `model.embed.weight`,
  `norm.weight` → `model.norm.weight`, `head.weight` → `lm_head.weight`).
* **Cache.** KV-source layers get a [`DeepseekV41CSACache`] layer (sliding-window ring + partial-group buffer +
  shared compressed KV / indexer keys); consumer layers read the source's cache through a per-forward `shared` dict.
  `DynamicCache(config=…)` builds the right layers from `config.layer_types` — the new
  `"shared_compressed_attention"` layer type is registered with the cache and masking utilities.
* **Group state across calls.** The compressor buffers partial groups, so chunked prefill and decode are exact:
  a prompt fed in chunks produces the same logits as a one-shot prefill, including when a chunk boundary lands
  inside a group.
* **Top-k ties.** The indexer's per-head scores are ReLU-rectified, so keys every head scores negatively tie at
  exactly 0.0 and `torch.topk` may order them by layout. Like DeepSeek-V3.2, cross-backend output-equivalence
  guarantees do not hold for this model; the equivalence tests in `tests/models/deepseek_v41` select all visible
  blocks to stay deterministic.
* **Stale indexer keys on incomplete-group decode steps (reference deviation).** The reference reassigns the
  shared indexer key cache only when a latent is emitted, but scores against it unconditionally: on decode steps
  where no group completes, a layer scores its queries against the *last publisher's* keys — typically layer 20's,
  since it owns the last index-source slot. This implementation (and vLLM, MLX and llama.cpp) instead always
  publishes the owner layer's running key cache, so those steps read the correct owner keys. The parity harness is
  blind to the difference (its tiny config saturates the top-k: `index_topk` exceeds the compressed length), so
  this is a documented deviation, not an asserted equivalence.

## DeepseekV41Config

[[autodoc]] DeepseekV41Config

## DeepseekV41TextConfig

[[autodoc]] DeepseekV41TextConfig

## DeepseekV41TextModel

[[autodoc]] DeepseekV41TextModel
    - forward

## DeepseekV41ForCausalLM

[[autodoc]] DeepseekV41ForCausalLM
    - forward
