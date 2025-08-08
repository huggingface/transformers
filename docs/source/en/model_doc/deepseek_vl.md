<!--Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.

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
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeepseekVL

[Deepseek-VL](https://arxiv.org/abs/2403.05525) was introduced by the DeepSeek AI team. It is a vision-language model (VLM) designed to process both text and images for generating contextually relevant responses. The model leverages [LLaMA](./llama) as its text encoder, while [SigLip](./siglip) is used for encoding images.

You can find all the original Deepseek-VL checkpoints under the [DeepSeek-community](https://huggingface.co/deepseek-community) organization.

> [!TIP]
> Click on the Deepseek-VL models in the right sidebar for more examples of how to apply Deepseek-VL to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="deepseek-community/deepseek-vl-1.3b-chat",
    device=0,
    dtype=torch.float16
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            { "type": "text", "text": "Describe this image."},
        ]
    }
]

pipe(text=messages, max_new_tokens=20, return_full_text=False)
```
</hfoption>

<hfoption id="AutoModel">

```py
import torch
from transformers import DeepseekVLForConditionalGeneration, AutoProcessor

model = DeepseekVLForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-1.3b-chat",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")

messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }

]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, dtype=model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
```
</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [torchao](../quantization/torchao) to only quantize the weights to int4.

```python
import torch
from transformers import TorchAoConfig, DeepseekVLForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig(
    "int4_weight_only",
    group_size=128
)

model = DeepseekVLForConditionalGeneration.from_pretrained(
    "deepseek-community/deepseek-vl-1.3b-chat",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)
```
### Notes

- Do inference with multiple images in a single conversation.
    ```py
    import torch
    from transformers import DeepseekVLForConditionalGeneration, AutoProcessor

    model = DeepseekVLForConditionalGeneration.from_pretrained(
        "deepseek-community/deepseek-vl-1.3b-chat",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    processor = AutoProcessor.from_pretrained("deepseek-community/deepseek-vl-1.3b-chat")

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s the difference between"},
                    {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
                    {"type": "text", "text": " and "},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"}
                ]
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"},
                    {"type": "text", "text": "What do you see in this image?"}
                ]
            }
        ]
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        padding=True,
        truncation=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text)
    ```

## DeepseekVLConfig

[[autodoc]] DeepseekVLConfig

## DeepseekVLProcessor

[[autodoc]] DeepseekVLProcessor

## DeepseekVLImageProcessor

[[autodoc]] DeepseekVLImageProcessor

## DeepseekVLImageProcessorFast

[[autodoc]] DeepseekVLImageProcessorFast

## DeepseekVLModel

[[autodoc]] DeepseekVLModel
    - forward

## DeepseekVLForConditionalGeneration

[[autodoc]] DeepseekVLForConditionalGeneration
    - forward
