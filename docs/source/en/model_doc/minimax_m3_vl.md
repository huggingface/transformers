<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-06-03.*


# MiniMax-M3-VL

## Overview

MiniMax-M3-VL is the vision-language member of the MiniMax-M3 family. It pairs a CLIP-style vision tower (Conv3d patch embedding with 3D rotary position embeddings) with the MiniMax-M3 text backbone, a mixed dense/sparse Mixture-of-Experts decoder that uses SwiGLU-OAI gated experts and a lightning indexer for block-sparse attention.

## Architecture

MiniMax-M3-VL is a CLIP-style vision tower joined to the MiniMax-M3 text backbone by a small GELU projector. The
text backbone is a Mixtral-style decoder with **plain residual connections** (no hyper-connections); its layers
vary along two independent per-layer axes — the MLP (`config.moe_layer_freq[i]`) and the attention
(`config.layer_types[i]`).

### Mixed dense/sparse MoE decoder

`config.moe_layer_freq[i]` selects layer `i`'s MLP:

* `0` — a dense [`MiniMaxM3VLDenseMLP`] at `dense_intermediate_size`.
* `1` — a sparse [`MiniMaxM3VLSparseMoeBlock`]: a [`MiniMaxM3VLTopKRouter`] routes the top-`num_experts_per_tok`
  of `num_local_experts` experts, scaled by `routed_scaling_factor`, with a single shared expert
  (`n_shared_experts` at `shared_intermediate_size`) running on every token in parallel.

The router scores experts with **`sigmoid`** (not softmax) and adds an auxiliary-loss-free `e_score_correction_bias`
*before* the top-k argmax, so the bias steers *which* experts are chosen without flowing gradients (DeepSeek's
`noaux_tc` trick); the chosen experts' sigmoid weights are then renormalized to sum to 1. Both dense and routed
experts use the **SwiGLU-OAI** activation — a clamped, sigmoid-gated GLU with a `+1` shift on the up branch:

```python
gate = gate.clamp(max=swiglu_limit)
up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
out = (up + 1.0) * (gate * torch.sigmoid(gate * swiglu_alpha))  # swiglu_alpha=1.702, swiglu_limit=7.0
```

### Block-sparse attention (Lightning Indexer)

Every layer is GQA (`num_key_value_heads = 4`) with per-head Gemma-style QK-norm and **partial RoPE** on the first
`rotary_dim` channels. `config.layer_types[i]` then picks `"full_attention"` (dense causal) or
`"minimax_m3_sparse"`, where a [`MiniMaxM3VLIndexer`] decides, per query, which keys the main attention may see.

The indexer is purely a *selection* branch — a small `index_n_heads`-head dot-product scorer with no value
projection and no residual output of its own (like DeepSeek-V4's indexer). It scores every key, then **max-pools
those per-key scores into blocks of `index_block_size` keys**, so selection happens at the granularity of a *block
of keys*: per query it keeps the top-`index_topk_blocks` key blocks plus the always-on `index_local_blocks`
local-window block (under block-level causality), broadcasts the per-block `0`/`-inf` verdict back onto every key in
the block, and re-applies token-level causality inside the diagonal block. The result is a `[B, 1, S_q, S_k]`
additive bias summed onto the causal mask, so the expensive main attention only touches the selected key blocks.

<img alt="MiniMax M3 Lightning Indexer mask" src="./minimax_m3_vl_indexer_mask.svg" />

### KV caching

Sparse layers cache the indexer's keys alongside the main KV. [`MiniMaxM3VLSparseCacheLayer`] (dynamic) and
[`MiniMaxM3VLSparseStaticCacheLayer`] (static / `torch.compile`) both register under
`layer_type = "minimax_m3_sparse"`, so `DynamicCache(config=text_config)` / `StaticCache` automatically pick the
sparse layer for each `layer_types[i] == "minimax_m3_sparse"` index — the same dispatch trick as DeepSeek-V4's
`DeepseekV4CSACache`. The indexer tracks its own `idx_cumulative_length` because it writes its keys *before* the
main attention writes its KV in each forward.

### Vision tower

A [`MiniMaxM3VLVisionModel`]: a `Conv3d` patch embedding over flattened `[N_patches, C·T·P·P]` input, a stack of
CLIP-style encoder layers carrying a **3D rotary** position embedding (time / height / width bands), and no
post-encoder norm — the last layer feeds the projector directly. A [`MiniMaxM3VLPatchMerger`] groups
`spatial_merge_size²` patches into the channel dim before the 2-layer GELU [`MiniMaxM3VLMultiModalProjector`] maps
vision features into the text hidden size.

### Differences from DeepSeek-V4

Both models share the *selection-only* lightning-indexer idea and an aux-loss-free MoE gate, but the surrounding
architecture differs substantially:

| Aspect | DeepSeek-V4 | MiniMax-M3-VL |
| --- | --- | --- |
| Modality | text-only | vision + video + text (CLIP-style tower, Conv3d patches, 3D RoPE) |
| Sparse attention | indexer selects top-k of a **compressed** KV pool (CSA `m=4` / HCA `m'=128`); mask is right-padded with compressor columns | indexer selects top-k **blocks of the raw keys** (block-sparse); bias is `[B, 1, S, S]` over the real key axis — no KV compression, no compressor cache |
| Attention layer types | 3-way (`sliding` / `compressed_sparse` / `heavily_compressed`) | 2-way (`full_attention` / `minimax_m3_sparse`) |
| Attention backbone | shared K=V MQA (1 KV head), grouped low-rank output projection, per-head learnable attention **sink**, extra sliding-window KV branch | plain GQA (4 KV heads) + per-head Gemma QK-norm + partial RoPE; **no** sink, no grouped output proj, no sliding branch |
| Residual stream | Manifold-Constrained Hyper-Connections (mHC), `hc_mult` parallel streams | plain residual connections (Mixtral-style) |
| MoE gate affinity | Sqrt(Softplus(·)) | Sigmoid(·) |
| MoE bootstrap | first layers are static `hash_moe` (frozen `tid2eid` lookup) | none — dense/MoE split is purely `moe_layer_freq` |
| Expert activation | clamped SwiGLU | SwiGLU-OAI (`(up + 1) · gate · sigmoid(gate · swiglu_alpha)`, `swiglu_alpha=1.702`) |

## Usage examples

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch


model = AutoModelForImageTextToText.from_pretrained(
    "MiniMaxAI/MiniMax-M3-preview",
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-M3-preview")

image = Image.new("RGB", (672, 672), (127, 127, 127))
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## MiniMaxM3VLConfig

[[autodoc]] MiniMaxM3VLConfig

## MiniMaxM3VLTextConfig

[[autodoc]] MiniMaxM3VLTextConfig

## MiniMaxM3VLVisionConfig

[[autodoc]] MiniMaxM3VLVisionConfig

## MiniMaxM3VLProcessor

[[autodoc]] MiniMaxM3VLProcessor

## MiniMaxM3VLImageProcessorFast

[[autodoc]] MiniMaxM3VLImageProcessorFast

## MiniMaxM3VLVideoProcessor

[[autodoc]] MiniMaxM3VLVideoProcessor

## MiniMaxM3VLVisionModel

[[autodoc]] MiniMaxM3VLVisionModel
    - forward

## MiniMaxM3VLTextModel

[[autodoc]] MiniMaxM3VLTextModel
    - forward

## MiniMaxM3VLModel

[[autodoc]] MiniMaxM3VLModel
    - forward

## MiniMaxM3VLForCausalLM

[[autodoc]] MiniMaxM3VLForCausalLM
    - forward

## MiniMaxM3VLForConditionalGeneration

[[autodoc]] MiniMaxM3VLForConditionalGeneration
    - forward
