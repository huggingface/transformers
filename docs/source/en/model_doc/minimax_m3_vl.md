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
### Block-sparse attention (Lightning Indexer)

Every layer is GQA (`num_key_value_heads = 4`) with per-head QK-norm and **partial RoPE** on the first
`rotary_dim`. `config.layer_types[i]` then picks `"full_attention"` (dense causal) or
`"minimax_m3_sparse"`, where a [`MiniMaxM3VLIndexer`] decides, per query, which block of keys the main attention may see.

The indexer scores every key, then **max-poolsthose per-key scores into blocks of `index_block_size` keys**, so selection happens at the granularity of a *block
of keys*: per query it keeps the top-`index_topk_blocks` key blocks plus the always-on `index_local_blocks`
local-window block (under block-level causality), broadcasts the per-block `0`/`-inf` choice back onto every key in
the block. The result is a `[B, 1, S_q, S_k]` additive bias summed onto the causal mask. 
Theoretically this means that the attention is only computed over the selected blocks of keys, but `transformers` does not support the kernels that compute this efficiently! 
We are adding it to `kernels` asap!

<img alt="MiniMax M3 Lightning Indexer mask" src="./minimax_m3_vl_indexer_mask.svg" />


### Vision tower

A [`MiniMaxM3VLVisionModel`]: a `Conv3d` patch embedding over flattened `[N_patches, C·T·P·P]` input, a stack of
CLIP-style encoder layers carrying a **3D rotary** position embedding (time / height / width bands). A [`MiniMaxM3VLPatchMerger`] groups
`spatial_merge_size²` patches into the channel dim before the 2-layer GELU [`MiniMaxM3VLMultiModalProjector`] maps vision features into the text hidden size.

## Usage examples

The example below runs the model on a real image loaded with [`~transformers.image_utils.load_image`].

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


model = AutoModelForImageTextToText.from_pretrained(
    "MiniMaxAI/MiniMax-M3-preview", dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-M3-preview")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
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

### Apple example

This example asks the model about an image of apples, again loading a real image with
[`~transformers.image_utils.load_image`].

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


model = AutoModelForImageTextToText.from_pretrained(
    "MiniMaxAI/MiniMax-M3-preview", dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-M3-preview")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "How many apples are in this image, and what color are they?"},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Fastest inference configuration

| ctx | SDPA decode | MSA decode | MSA decode adv. | SDPA prefill | MSA prefill | MSA prefill adv. |
| --: | ----------: | ---------: | --------------: | -----------: | ----------: | ---------------: |
|  2K |  27.8 tok/s |       31.0 |            +12% |       303 ms |      257 ms |            1.18× |
|  4K |  23.4 tok/s |       30.5 |            +30% |       684 ms |      460 ms |            1.49× |
|  8K |  17.8 tok/s |       29.6 |            +66% |      1906 ms |      976 ms |            1.95× |
| 16K |  12.0 tok/s |       27.6 |           +130% |      6110 ms |     2344 ms |            2.61× |

The checkpoint ships in native MXFP8. For **decode throughput**, the fastest validated configuration is
**bf16 (dequantized at load) + the MSA block-sparse attention kernel + tensor & expert parallelism + a
`reduce-overhead` cudagraph compile** — roughly **31 tok/s** decode on 8×B200 at a 2048-token prefill.

Keeping the weights in **native FP8 is a memory-footprint option only — it is never faster on this setup**.
The FP8 Triton experts/linear kernels lower as opaque inductor fallback kernels that cudagraph cannot
capture on the hot expert path, so native-FP8 decode measured ~4.2 tok/s (≈7× slower than the bf16 path)
even under `torch.compile(fullgraph=True)`. Use FP8 only when the bf16 weights do not fit.

| config (sdpa baseline, TP+EP, 2048-token prefill, 8×B200) | decode |
|---|---|
| bf16 dequantize-at-load + **MSA** + compile/cudagraph | **~31 tok/s** |
| bf16 dequantize-at-load + sdpa + compile/cudagraph | ~28 tok/s |
| native FP8 + compile/cudagraph | ~4 tok/s (memory-only, not for speed) |

Dequantizing to bf16 only fits with even sharding across GPUs (TP/EP), not with `device_map="auto"`
(pipeline placement OOMs at load). Launch one process per GPU with `torchrun`:

```bash
torchrun --nproc_per_node=8 fastest_m3_vl.py
```

```python
# fastest_m3_vl.py
import os, sys
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    CompileConfig,
    FineGrainedFP8Config,
)
from transformers.distributed import DistributedConfig

# The indexer feeds SDPA an additive float mask; the cuDNN SDP backend segfaults on it (B200).
torch.backends.cuda.enable_cudnn_sdp(False)

model = AutoModelForImageTextToText.from_pretrained(
    "MiniMaxAI/MiniMax-M3-preview",
    dtype=torch.bfloat16,
    # Dequantize the native MXFP8 weights to bf16 at load (the speed win); needs even TP/EP sharding.
    quantization_config=FineGrainedFP8Config(dequantize=True),
    tp_plan="auto",
    distributed_config=DistributedConfig(enable_expert_parallel=True),
    attn_implementation="kernels-staging/msa@v0",  # MSA block-sparse attention kernel
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M3-preview")
messages = [{"role": "user", "content": "Summarize the history of computing."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(f"cuda:{os.environ.get('LOCAL_RANK', '0')}")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
    # Static cache + reduce-overhead cudagraph capture is what pushes decode to ~31 tok/s.
    cache_implementation="static",
    compile_config=CompileConfig(mode="reduce-overhead", fullgraph=True),
)
if int(os.environ.get("RANK", "0")) == 0:
    print(tokenizer.decode(generated_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# cudagraph-captured NCCL collectives deadlock the NCCL/CUDA destructors at teardown; the output is
# already produced, so hard-exit to skip the hanging cleanup.
if dist.is_initialized():
    sys.stdout.flush()
    os._exit(0)
```

## MiniMaxM3VLConfig

[[autodoc]] MiniMaxM3VLConfig

## MiniMaxM3VLTextConfig

[[autodoc]] MiniMaxM3VLTextConfig

## MiniMaxM3VLVisionConfig

[[autodoc]] MiniMaxM3VLVisionConfig

## MiniMaxM3VLProcessor

[[autodoc]] MiniMaxM3VLProcessor

## MiniMaxM3VLImageProcessor

This is a standalone (non-modular) image processor: it shares the patch-flattening idea of [`Qwen2VLImageProcessor`]
but does not inherit from it because the two diverge in ways that touch most of the class. The resize budget is driven by
a `max_pixels` attribute and a `{"height", "width"}` `size` rather than Qwen's `shortest_edge`/`longest_edge` scheme; the
`smart_resize` helper clamps the initial rounding with `max(factor, ...)`; and `_preprocess` performs real temporal
handling (5D patches, last-frame repeat to fill `temporal_patch_size`, and a `grid_t` dimension) instead of Qwen's
`grid_t = 1` + expand. Mapping to or subclassing Qwen would therefore change behavior or require overriding nearly
everything, so the processor is kept on its own.

[[autodoc]] MiniMaxM3VLImageProcessor

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

## MiniMaxM3SparseForConditionalGeneration

[[autodoc]] MiniMaxM3SparseForConditionalGeneration
    - forward
