<!--Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

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
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Qwen2.5-VL

[Qwen2.5-VL](https://huggingface.co/papers/2502.13923) is a multimodal vision-language model, available in 3B, 7B, and 72B parameters, pretrained on 4.1T tokens. The model introduces window attention in the ViT encoder to accelerate training and inference, dynamic FPS sampling on the spatial and temporal dimensions for better video understanding across different sampling rates, and an upgraded MRoPE (multi-resolutional rotary positional encoding) mechanism to better capture and learn temporal dynamics.


You can find all the original Qwen2.5-VL checkpoints under the [Qwen2.5-VL](https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5) collection.

> [!TIP]
> Click on the Qwen2.5-VL models in the right sidebar for more examples of how to apply Qwen2.5-VL to different vision and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline
pipe = pipeline(
    task="image-text-to-text",
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    device=0,
    dtype=torch.bfloat16
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
pipe(text=messages,max_new_tokens=20, return_full_text=False)

```
</hfoption>

<hfoption id="AutoModel">

```py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
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
).to("cuda")

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
from transformers import TorchAoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor

quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

```
### Notes

- Use Qwen2.5-VL for video inputs by setting `"type": "video"` as shown below.
    ```python
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "/path/to/video.mp4"},
                {"type": "text", "text": "What happened in the video?"},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        conversation,
        video_fps=1,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text)
    ```
- Use Qwen2.5-VL for a mixed batch of inputs (images, videos, text). Add labels when handling multiple images or videos for better reference
 as show below.
    ```python
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": "Hello, how are you?"}
            ]
        },
        {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking. How can I assist you today?"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you describe these images and video?"}, 
                {"type": "image"}, 
                {"type": "image"}, 
                {"type": "video"}, 
                {"type": "text", "text": "These are from my vacation."}
            ]
        },
        {
            "role": "assistant",
            "content": "I'd be happy to describe the images and video for you. Could you please provide more context about your vacation?"
        },
        {
            "role": "user",
            "content": "It was a trip to the mountains. Can you see the details in the images and video?"
        }
    ]
    
    # default:
    prompt_without_id = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?<|vision_start|><|image_pad|><|vision_end|><|vision_start|><|image_pad|><|vision_end|><|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
    
    
    # add ids
    prompt_with_id = processor.apply_chat_template(conversation, add_generation_prompt=True, add_vision_id=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>Hello, how are you?<|im_end|>\n<|im_start|>assistant\nI'm doing well, thank you for asking. How can I assist you today?<|im_end|>\n<|im_start|>user\nCan you describe these images and video?Picture 2: <|vision_start|><|image_pad|><|vision_end|>Picture 3: <|vision_start|><|image_pad|><|vision_end|>Video 1: <|vision_start|><|video_pad|><|vision_end|>These are from my vacation.<|im_end|>\n<|im_start|>assistant\nI'd be happy to describe the images and video for you. Could you please provide more context about your vacation?<|im_end|>\n<|im_start|>user\nIt was a trip to the mountains. Can you see the details in the images and video?<|im_end|>\n<|im_start|>assistant\n'
    ```

- Use the `min_pixels` and `max_pixels` parameters in [`AutoProcessor`] to set the resolution.

    ```python
    min_pixels = 224*224
    max_pixels = 2048*2048
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    ```
    
    Higher resolution can require more compute whereas reducing the resolution can save memory as follows:
    
    ```python
    min_pixels = 256*28*28
    max_pixels = 1024*28*28 
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    ```
## Qwen2_5_VLConfig

[[autodoc]] Qwen2_5_VLConfig

## Qwen2_5_VLTextConfig

[[autodoc]] Qwen2_5_VLTextConfig

## Qwen2_5_VLProcessor

[[autodoc]] Qwen2_5_VLProcessor

## Qwen2_5_VLTextModel

[[autodoc]] Qwen2_5_VLTextModel
    - forward

## Qwen2_5_VLModel

[[autodoc]] Qwen2_5_VLModel
    - forward

## Qwen2_5_VLForConditionalGeneration

[[autodoc]] Qwen2_5_VLForConditionalGeneration
    - forward
