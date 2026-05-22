<!--Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Cosmos3 Omni

[Cosmos3](https://huggingface.co/nvidia/Cosmos3-Nano) is a mixture-of-transformers (MoT) Vision Foundation Model from NVIDIA, composed of a *Reasoner* tower and a *Generator* tower. The two towers share the same input embedding and visual encoder but use disjoint MoT experts for understanding vs. generation, plus cross-modal adapters (`llm2vae`, `llm2sound`, `llm2action`, etc.) that connect the language model to image / audio / action heads.

The transformers integration loads **only the Reasoner tower** from a unified Cosmos3 checkpoint. The Reasoner is architecturally identical to [Qwen3-VL](./qwen3_vl) — `Cosmos3ReasonerForConditionalGeneration` is a thin subclass of `Qwen3VLForConditionalGeneration`. Loading is driven by two transformations registered automatically when `model_type` is `"cosmos3_omni"`:

1. The checkpoint's flat namespaces are re-targeted to Qwen3-VL's nested layout: `model.<*>` → `model.language_model.<*>` and `blocks.*` / `merger.*` / `patch_embed.*` / `pos_embed.*` / `deepstack_merger_list.*` → `model.visual.<*>`.
2. Generator / sound / action parameters (`*_moe_gen`, `llm2vae`, `vae2llm`, `time_embedder`, `llm2sound`, `sound2llm`, `sound_modality_embed`, `llm2action`, `action2llm`, `action_modality_embed`) are skipped on load.

## Usage

```python
import torch
from transformers import AutoProcessor, Cosmos3ReasonerForConditionalGeneration

model = Cosmos3ReasonerForConditionalGeneration.from_pretrained(
    "nvidia/Cosmos3-Nano",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained("nvidia/Cosmos3-Nano")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Caption the image in detail."},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
output = processor.batch_decode(
    [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output[0])
```

## Cosmos3Config

[[autodoc]] Cosmos3Config

## Cosmos3Model

[[autodoc]] Cosmos3Model
    - forward
    - get_video_features
    - get_image_features

## Cosmos3ReasonerForConditionalGeneration

[[autodoc]] Cosmos3ReasonerForConditionalGeneration
    - forward
    - get_video_features
    - get_image_features
