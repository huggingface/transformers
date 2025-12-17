<!--Copyright 2025 Perceptron, Inc. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-12.*
*This model was added to Hugging Face Transformers in 2025.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Isaac

Isaac is Perceptron's vision-language model (VLM) that pairs a SigLIP2 vision encoder with a Qwen3 decoder-only stack. The
architecture is designed for efficient long-context multimodal interactions, and supports interleaving images with
text. The vision encoder has variable-resolution capability and with optional pixel shuffle to merge
neighboring patches before they reach the decoder, which keeps the KV-cache and compute requirements manageable on long
conversations. Text and vision tokens are unified via the [`TensorStream`](https://github.com/perceptron-ai-inc/perceptron/tree/main/src/perceptron/tensorstream) abstraction so
that modal boundaries, spatial coordinates, and rescaling parameters are preserved throughout the model stack. For more information, refer to the [technical report](https://github.com/perceptron-ai-inc/perceptron/blob/main/papers/isaac_01.pdf).

Key implementation notes:

- **Packed vision attention** – `IsaacVisionEncoder` keeps track of per-image patch lengths and uses specialized attention
  kernels with custom `AttentionMaskConverter` utilities so the decoder only applies attention to real patches while supporting
  both FlashAttention and SDPA.
- **TensorStream-first pipeline** – `IsaacProcessor` converts chat templates into multimodal streams where every image gets a
  dedicated event with spatial metadata. `IsaacModel` can embed that stream directly (using `embed_stream`) and automatically
  derive multi-dimensional RoPE coordinates, so you only need to provide the `tensor_stream` during the first decoding step.
- **Fast image pre-processing** – `IsaacImageProcessorFast` solves for the closest resolution that fits within the requested context.

Isaac checkpoints are distributed under Perceptron's Non-Production license; please review the license that ships with the
weights before using them in commercial settings.

## Usage example

`IsaacProcessor` expects that every `<image>` token in the rendered prompt has a
matching image. The processor returns both standard tokenized inputs and a `TensorStream`. You should pass the stream to the
model (only the first generation step requires it) alongside the regular tensors.

```py
import torch
from PIL import Image
from transformers import AutoProcessor, IsaacForConditionalGeneration

model_id = "Perceptron/isaac-base"
processor = AutoProcessor.from_pretrained(model_id)
model = IsaacForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

images = [Image.open("chart.png"), Image.open("panel.jpg")]
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": images[0]},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": "Compare the two figures and explain what changed."},
        ],
    }
]

# Render the chat template to text so we can pass text+images together.
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# IsaacProcessor builds TensorStream events internally when both text and images are provided.
batch = processor(text=prompt, images=images, return_tensors="pt")

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        tensor_stream=tensor_stream,
        max_new_tokens=256,
        temperature=0.2,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

response = processor.post_process_image_text_to_text(
    generated,
    skip_special_tokens=True,
)[0]
print(response)
```

## IsaacConfig

[[autodoc]] IsaacConfig

## IsaacModel

[[autodoc]] IsaacModel
    - forward

## IsaacForConditionalGeneration

[[autodoc]] IsaacForConditionalGeneration
    - forward

## IsaacProcessor

[[autodoc]] IsaacProcessor

## IsaacImageProcessorFast

[[autodoc]] IsaacImageProcessorFast
