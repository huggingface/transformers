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
*This model was contributed to Hugging Face Transformers on 2026-08-16.*

# Step3p7 (Step-3.7-Flash)

## Overview

Step-3.7-Flash was proposed in [Step 3.7 Flash](https://static.stepfun.com/blog/step-3.7-flash/) by StepFun. It is a 198B-parameter sparse Mixture-of-Experts vision-language model, pairing a 196B-parameter MoE language backbone with a 1.8B-parameter vision encoder for native image understanding.

## Architecture

StepFun hasn't published a technical report for Step-3.7-Flash, so the details below are drawn from the released checkpoint's configuration rather than a paper.

- **Sparse MoE decoder**: all but the first 3 decoder layers route through a MoE block of 288 routed experts (top-8 per token) plus a single shared expert. The router scores experts with a sigmoid and a learned per-expert bias instead of an auxiliary load-balancing loss, the same strategy as [DeepSeek-V3](./deepseek_v3).
- **Gated attention**: each attention layer adds an extra projection whose sigmoid output gates the attention output per head, before the output projection — the same *Gated Attention* mechanism used in [Qwen3-Next](./qwen3_next). A subset of layers use fewer heads and a sliding window instead of full attention.
- **Multi-token prediction**: some checkpoints ship extra decoder layers trained for multi-token prediction, which [`~GenerationMixin.generate`] can use for speculative decoding via `use_mtp=True`.
- **Vision encoder**: a SigLIP-style ViT with 2-D rotary position embeddings and a learned per-layer scale on the attention and MLP branches. Its output is downsampled 4x by two stride-2 convolutions before a linear projector maps it into the text model's hidden size.
- **Dynamic image tiling**: instead of a fixed tile grid, the image processor picks its tiling window from each image's own aspect ratio, producing one downscaled global view plus zero or more local high-resolution crops per image.

## Usage example

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


model = AutoModelForImageTextToText.from_pretrained(
    "stepfun-ai/Step-3.7-Flash", dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("stepfun-ai/Step-3.7-Flash")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Step3p7Config

[[autodoc]] Step3p7Config

## Step3p7VisionConfig

[[autodoc]] Step3p7VisionConfig

## Step3p7TextConfig

[[autodoc]] Step3p7TextConfig

## Step3p7ImageProcessor

[[autodoc]] Step3p7ImageProcessor

## Step3p7Processor

[[autodoc]] Step3p7Processor

## Step3p7VisionModel

[[autodoc]] Step3p7VisionModel
    - forward

## Step3p7TextModel

[[autodoc]] Step3p7TextModel
    - forward

## Step3p7Model

[[autodoc]] Step3p7Model
    - forward

## Step3p7ForConditionalGeneration

[[autodoc]] Step3p7ForConditionalGeneration
    - forward
