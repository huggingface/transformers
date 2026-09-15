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
*This model was contributed to Hugging Face Transformers on 2026-09-13.*

# DeepSeek-V4.1

[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai) is the next MoE language model from DeepSeek. V4.1 keeps
V4's shared-K=V latent attention, Manifold-Constrained Hyper-Connections (mHC) and grouped low-rank output
projection, and replaces the fixed three-type attention schedule with a **ratio-scheduled** compressed-attention
design (CSA2), adds a **two-level indexer** (a candidate pre-filter constrains every later indexer), and introduces
the **engram** — deterministic n-gram hash lookups injected into the residual stream at a few layers.

The text backbone is available as [`DeepseekV41TextModel`] and [`DeepseekV41ForCausalLM`]. Image-conditioned
generation uses [`DeepseekV41ForConditionalGeneration`], which combines DeepSeek-ViT, the aligner and the text
backbone. The MTP draft head (DSpark) remains unsupported and its checkpoint keys are ignored on load.

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

The residual stream is carried as `hc_mult` parallel copies. Each attention / FFN site (`attn_hc` / `ffn_hc`, a
`DeepseekV41HyperConnection` with the same `fn` / `base` / `scale` parametrization as V4) mixes the streams with a
row-normalized (Sinkhorn-projected) mixing matrix; unlike V4, the site's pre-mix is consumed one site *ahead*
(single-pass mHC), so the pipeline crosses layer boundaries exactly as in the released implementation. The mHC
sites and the attention sinks stay in `float32` — matching the released checkpoint's dtypes
(`_keep_in_fp32_modules_strict`).

### Engram

At `config.engram_layer_ids`, `DeepseekV41Engram` adds n-gram hash-table lookups into the residual stream: each
position is hashed with its `engram_max_ngram_size - 1` predecessor tokens (multiplied by per-layer random
multipliers, folded into prime-sized buckets), and each of the `engram_n_heads` heads reads one embedding row per
n-gram size. The hash is a pure function of `(tokenizer, config)` — no learned table is involved in the id mapping.

The state therefore needs the **tokenizer** to build its compressed token map. `from_pretrained` binds the
checkpoint's own tokenizer after loading (nothing is fetched in the forward pass); bind one explicitly when the
model was built any other way or the checkpoint ships no tokenizer:

```python
model.model.bind_tokenizer(tokenizer)  # explicit
# or: model = DeepseekV41ForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
#    (the repo's tokenizer is bound at load time)
```

Tokens masked out of n-grams (image spans) are hashed as a DEAD sentinel; look-back stops at them, so an n-gram
never spans one. Across forward calls (chunked prefill, decode) the look-back lives on the cache — the first
`shared_compressed_attention` layer (`DeepseekV41CSACache` via the `DeepseekV41EngramHistoryLayer` mixin) — so it
follows the KV through beam reorders, `num_return_sequences` and `reset`.

### Mixture of experts

The routed experts are one fused `DeepseekV41Experts` module per layer (`gate_up_proj` `[E, 2·inter, hidden]` with
the gate rows first, `down_proj` `[E, hidden, inter]`), dispatched through the shared experts interface: the default
`experts_implementation` is `grouped_mm` (falls back to `batched_mm` / `eager`; `from_pretrained(...,
experts_implementation="eager")` or `model.set_experts_implementation(...)` select one). The eager path is the
reference math — fp32 clamped SwiGLU (`up` clamped on both sides, `gate` from above), routing weight applied to the
activation *before* the down projection, fp32 accumulation over experts and the shared expert (`shared_experts`, a
clamped `DeepseekV41MLP`). The router (`gate`) keeps its expert-selection correction biases as fp32 buffers
(`e_score_correction_bias`, and `e_score_correction_bias_vl` for image-span tokens); `output_router_logits=True`
records the pre-activation gate logits per layer and adds the Mixtral load-balancing auxiliary loss
(`router_aux_loss_coef`).

### Vision tower and image spans

[`DeepseekV41VisionModel`] applies bidirectional attention independently to each image's row-major patch grid.
It uses axial 2D RoPE, RMSNorm and SwiGLU. The aligner groups patches into channel-major 3×3 windows, zero-pads
incomplete edge windows, and projects them into the text hidden size through a two-layer GELU MLP.

[`DeepseekV41ImageProcessor`] is the torchvision backend; [`DeepseekV41ImageProcessorPil`] is the PIL backend.
Both implement the reference resize plan, RGB conversion, gray padding, normalization and patch layout.
The model receives `pixel_values` as packed patch rows and `image_grid_thw` in image order, with rows `[1, h, w]`.
Videos are not supported by these image-only classes.

[`DeepseekV41Processor`] expands one image placeholder into
`[IMAGE_START] + ([IMAGE] * width + [IMAGE_NEWLINE]) * height + [IMAGE_END]`.
All of these positions carry the same image token ID: **129264**, spelled **`<｜deepseek_image｜>`** in the
released tokenizer. The model reconstructs delimiter and image-feature positions from each image grid.
Image spans use the vision routing bias and are excluded from Engram n-grams; they remain live tokens for attention.

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
FP4** (e2m1 nibbles in an int8 container, one ue8m0 scale per row per 32 fp4 channels), the compressor / indexer
projections, embeddings and head in bf16, and the mHC / sink / gate-bias parameters in fp32. `from_pretrained`
accepts this layout as-is:

* with `dequantize=True` (the default on CPU) every tensor is dequantized to the requested `dtype` — the fp8
  quantizer's `.scale` → `weight_scale_inv` mapping and its FP4-aware dequantize op run *before* the per-expert
  `w1` / `w3` / `w2` tensors are merged into the fused `gate_up_proj` / `down_proj`, and the per-row-scaled engram
  tables are dequantized into the embedding (the lookup stays a plain row gather);
* with `dequantize=False` (GPU) the attention projections become `FP8Linear` / `FP8GroupedLinear` (the grouped
  `o_a_proj`) and the routed experts an `FP8Experts` holding the packed int8 `gate_up_proj` / `down_proj` with their
  `[1, 32]` ue8m0 scales (`gate_up_proj_scale_inv` / `down_proj_scale_inv`). `FP8Experts` sizes that packed layout
  from `text_config.expert_dtype`, which the configuration loaders preserve from the released config.json's
  `quantization_config.expert_dtype = "fp4"`. The bf16 projections without a `.scale` (`compressor.kv_proj`,
  `compressor.gate_proj`, `indexer.k_proj`, `indexer.weights_proj`) are auto-skipped by the quantizer
  (`_keep_in_fp32_modules`), like V4's.

`DeepseekV41ForCausalLM` uses the composite [`DeepseekV41Config`]; its text configuration is unwrapped during
initialization. [`DeepseekV41TextModel`] uses [`DeepseekV41TextConfig`] and preserves the outer
`quantization_config` before extracting `text_config`, so the bare backbone also recognizes the released
quantized checkpoint. Flat text checkpoints and direct text-config construction remain supported.

## Usage

### Text-only inference

The released checkpoint (476 GiB on disk: fp8 attention, packed-fp4 routed experts, two ~98 GB fp8 engram tables)
loads through the standard path — no custom loader:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash", dtype="bfloat16", device_map="auto")
inputs = tok("Hello, my dog is cute", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))
```

`device_map="auto"` spreads the decoder layers over the available accelerators with the fp8 / fp4 weights kept
quantized (`dequantize=False` on CUDA) and runs them through the fp8 kernels. The two engram tables are
`_no_placement_params`: on accelerators that cannot hold a 98 GB table they stay in host RAM (the row gather runs
there and only the rows move), on ones that can they are placed like any other weight. On 8×H100 80 GB this
loads in ~6 minutes, uses ~41 GiB per GPU, and reproduces the eager reference token-for-token; a single process
spanning several devices is routed to the Triton fp8 kernels (the DeepGEMM path binds to one CUDA context), at
roughly 0.4–1 s per generated token.

The sibling-standard fast path is expert parallelism — one process per GPU under `torchrun`, `DistributedConfig`
with `enable_expert_parallel=True`. It uses `base_model_ep_plan`: the routed experts are sharded along the expert
axis (`grouped_gemm`), the gate routes (`ep_router`), and the engram tables are `nn.Embedding`s sharded along the
embedding dim (`colwise_gather_output`, like Qwen4-Exp's n-gram table) so each rank holds `head_dim / tp_size`
channels of every row — whole 32-channel scale blocks, so the fp8 dequantization stays rank-local:

```python
import os
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V4.1-Flash",
    dtype="bfloat16",
    distributed_config=DistributedConfig(tp_size=int(os.environ["WORLD_SIZE"]), enable_expert_parallel=True),
)
```

```bash
torchrun --nproc-per-node 8 your_script.py
```

Per-rank memory under EP is the replicated attention (~17 GB fp8) + `experts / tp_size` (~36 GB on 8 ranks) +
`tables / tp_size` (~24 GB on 8 ranks): about 77 GB, which does not leave room on an 80 GB H100 — use 141 GB
H200 / 192 GB B200 parts. The native quantized engram layout requires `tp_size` to divide
`engram_head_dim / 32` (8 for the released model); 16-rank EP is not supported by this sharding plan.
Measured on 8×H200 (batch 1, greedy): load 432 s, 66 GiB per rank,
prefill 0.36 s for a short prompt and 0.70 s for 1.2k tokens, decode 327 ms/token — on par with `deepseek_v4`
(DeepSeek-V4-Flash: 373 ms/token on the same harness).

Long prompts: the compressed branch attends only the `index_topk` entries the indexer picks per query. Its
attention work uses the selected entries rather than the entire compressed cache, and indexer scoring uses
bounded query chunks; the KV caches themselves still grow with context length. Feed long prompts in chunks
(`model(chunk, past_key_values=cache)` carries partial groups and n-gram history) and let `generate` continue
from the cache; a needle-in-a-haystack
passcode was retrieved at 262k, 524k and 1,048k prompt tokens on 8×H200 (4096-token chunks: 754 / 595 / 394
tokens per second of prefill).

### Image-conditioned generation

The released repository does not include Hugging Face processor configuration files or a Jinja chat template.
Construct the processor from its vision config and tokenizer; do not assume `AutoProcessor.from_pretrained`
can recover absent assets. A processor saved with `save_pretrained` can subsequently be loaded with `AutoProcessor`.

Use the release's [Python prompt encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/encoding)
or `deepseek-recipe` for chat serialization. For the following example, download `encoding/encoding.py` from the
model repository and put that directory on `PYTHONPATH`:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash --include 'encoding/*' --local-dir ./deepseek-v41
export PYTHONPATH="./deepseek-v41/encoding:$PYTHONPATH"
```

```python
from PIL import Image
from encoding import encode_messages
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoTokenizer,
    DeepseekV41ImageProcessor,
    DeepseekV41Processor,
)

model_id = "deepseek-ai/DeepSeek-V4.1-Flash"
config = AutoConfig.from_pretrained(model_id)
vision = config.vision_config
tokenizer = AutoTokenizer.from_pretrained(model_id)
image_processor = DeepseekV41ImageProcessor(
    patch_size=vision.patch_size,
    downsample_ratio=vision.downsample_ratio,
    min_pixels=vision.min_pixels,
    max_image_tokens=vision.max_image_tokens,
    max_wh_ratio=vision.max_wh_ratio,
)
processor = DeepseekV41Processor(image_processor=image_processor, tokenizer=tokenizer)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="bfloat16", device_map="auto"
)
image = Image.open("image.png").convert("RGB")
prompt = encode_messages(
    [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "image.png"}},
    ]}],
    thinking_mode="chat",
)
inputs = processor(text=prompt, images=[image], add_special_tokens=False, return_tensors="pt").to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

Image spans must be included completely in the initial prefill. Later calls can continue text generation from
the returned cache. The performance and long-context measurements above were made on the **text-only** path;
they are not measurements of vision latency or full-checkpoint multimodal accuracy.

> [!NOTE]
> The **text backbone** runs eager attention only: its large head dimension, denominator sink and compressed
> branch require a custom attention path. The vision tower supports eager attention and SDPA independently.
> `StaticCache` / `QuantizedCache` are not supported either — the compressor's group state is dynamic
> (`DynamicCache`, the default, builds the right layers automatically).

## Implementation notes

* **Naming.** The modules follow the Transformers conventions (`self_attn.q_a_proj` / `kv_proj` / `o_a_proj` /
  `sinks`, `mlp.gate` / `experts.gate_up_proj` / `shared_experts.gate_proj`, `attn_hc.fn`, `embed_tokens`,
  `lm_head`); the released checkpoint's DeepSeek-native names (`attn.wq_a`, `ffn.experts.E.w1`, raw `hc_attn_fn`
  on the layer, top-level `embed` / `head`) are mapped by the `deepseek_v41_text` entry of
  `conversion_mapping.py` — a rename set plus the per-expert `w1` / `w3` → `gate_up_proj` (`MergeModulelist` +
  `Concatenate`) and `w2` → `down_proj` merges. The mapping is keyed by the text `model_type`, so it also applies
  to a bare [`DeepseekV41TextModel`]; the wrapper's `model.` prefix is the loader's. The engram's per-layer
  `engram.wkv` / `engram.{q,k}_weight` keep their names; its hash table moves from `layers.N.engram.embed.*` to
  the model-level `engram_tables.N.*` (a no-split [`DeepseekV41EngramEmbedding`] whose `weight` /
  `weight_scale_inv` are `_no_placement_params`: a no-split decoder layer that held the table would be split
  into parameter-level `device_map` entries, which carry no accelerate hooks). The fp8 quantizer's global
  `.scale` → `weight_scale_inv` rename is why the table's scales carry that name.
* **Recorded `hidden_states`.** The residual stream is `hc_mult` parallel copies, so `output_hidden_states`
  records the *collapsed* per-block inputs (each layer's `input_layernorm` input, plus the initial embedding
  collapse and the final normalized state) in the standard `[batch, seq, hidden]` shape.
* **Cache.** KV-source layers get a [`DeepseekV41CSACache`] layer (sliding-window ring + partial-group buffer +
  shared compressed KV / indexer keys); consumer layers read the source's cache through a per-forward `shared` dict.
  `DynamicCache(config=…)` builds the right layers from `config.layer_types` — the new
  `"shared_compressed_attention"` layer type is registered with the cache and masking utilities.
  CSA caches are not croppable: removing tokens cannot rewind the compressor's partial groups or emitted entries.
  Nonzero rollback requests are rejected before any cache layer is mutated; `crop(0)` retains trim-only behavior.
* **Padding and group state.** Compression groups contain only live tokens from each batch row. Their first-token
  RoPE positions, live-token counts and partial groups follow the row through cached calls and beam operations.
  Chunked prefill and decode use the same groups as one-shot prefill, even when a chunk ends inside a group or
  contains only padding for some rows. Floating-point reduction and hard top-k boundaries can still affect
  numerical equivalence.
* **Top-k ties.** The indexer's per-head scores are ReLU-rectified, so keys every head scores negatively tie at
  exactly 0.0 and `torch.topk` may order them by layout. Like DeepSeek-V3.2, cross-backend output-equivalence
  guarantees do not hold for this model; the equivalence tests in `tests/models/deepseek_v41` select all visible
  blocks to stay deterministic.
* **Stale indexer keys on incomplete-group decode steps (reference deviation).** The reference reassigns the
  shared indexer key cache only when a latent is emitted, but scores against it unconditionally: on decode steps
  where no group completes, a layer scores its queries against the *last publisher's* keys — typically layer 20's,
  since it owns the last index-source slot. This implementation republishes the owning layer's running cache.
  The same reference fix is proposed in [DeepSeek's model PR #12](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/12).
  [Discussion #41](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/discussions/41) provides a reduced-model CPU
  reproduction and states its kernel-adaptation and numerical-validation limits.

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

## DeepseekV41VisionConfig

[[autodoc]] DeepseekV41VisionConfig

## DeepseekV41VisionModel

[[autodoc]] DeepseekV41VisionModel
    - forward

## DeepseekV41Model

[[autodoc]] DeepseekV41Model
    - forward
    - get_image_features

## DeepseekV41ForConditionalGeneration

[[autodoc]] DeepseekV41ForConditionalGeneration
    - forward
    - get_image_features

## DeepseekV41ImageProcessor

[[autodoc]] DeepseekV41ImageProcessor
    - preprocess

## DeepseekV41ImageProcessorPil

[[autodoc]] DeepseekV41ImageProcessorPil
    - preprocess

## DeepseekV41Processor

[[autodoc]] DeepseekV41Processor

## DeepseekV41CSACache

[[autodoc]] DeepseekV41CSACache

## DeepseekV41EngramEmbedding

[[autodoc]] DeepseekV41EngramEmbedding

## DeepseekV41NgramHashState

[[autodoc]] DeepseekV41NgramHashState

## EngramLayout

[[autodoc]] EngramLayout
