<!--Copyright 2026 NAVER Corp. and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-30.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# HyperCLOVAX Vision V2

HyperCLOVAX Vision V2 is a multimodal vision-language model developed by NAVER. It combines the HyperCLOVAX language model backbone — based on the [Granite](./granite) architecture with optional post-norm (Peri-LN) layers for MuP scaling — with a [Qwen2.5-VL](./qwen2_5_vl) vision encoder. The model supports text, image, and video inputs and is capable of chain-of-thought reasoning via built-in thinking tokens (`<think>...</think>`).

You can find the original HyperCLOVAX-SEED-Think-32B checkpoint on the [naver-hyperclovax/HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) page.

> [!TIP]
> The `model_type` in the released checkpoint's `config.json` is `"vlm"`, while the Transformers implementation registers this model as `"hyperclovax_vision_v2"`. Due to this mismatch, loading via `AutoModel` or `AutoModelForCausalLM` is not supported. Use the model class directly as shown in the examples below.

The example below demonstrates how to generate text based on an image with [`HCXVisionV2ForConditionalGeneration`].

<hfoptions id="usage">
<hfoption id="Image input">

```python
import torch
from transformers import HCXVisionV2ForConditionalGeneration, HCXVisionV2Processor

model = HCXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = HCXVisionV2Processor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
<hfoption id="Video input">

```python
import torch
from transformers import HCXVisionV2ForConditionalGeneration, HCXVisionV2Processor

model = HCXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = HCXVisionV2Processor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "/path/to/video.mp4"},
            },
            {"type": "text", "text": "Describe this video."},
        ],
    },
]

# Use processor.tokenizer.apply_chat_template for video inputs.
# processor.apply_chat_template rewrites image_url to image before the
# template runs, which breaks HCX's extension-based video detection.
text = processor.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)
inputs = processor(
    text=text,
    videos=["/path/to/video.mp4"],
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

</hfoption>
</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to load the model in 4-bit.

```python
import torch
from transformers import BitsAndBytesConfig, HCXVisionV2ForConditionalGeneration, HCXVisionV2Processor

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = HCXVisionV2ForConditionalGeneration.from_pretrained(
    "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config,
)
processor = HCXVisionV2Processor.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
```

## Notes

- HyperCLOVAX Vision V2 uses a unique media input format. Both images and videos are specified using `{"type": "image_url", "image_url": {"url": "..."}}`. The processor and chat template distinguish images from videos by file extension (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`, `.m4v` are treated as video; everything else is treated as image).

    ```python
    # Image input
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}

    # Video input (identified by file extension)
    {"type": "image_url", "image_url": {"url": "/path/to/video.mp4"}}
    ```

    > [!WARNING]
    > Video input via `processor.apply_chat_template` is currently broken. Recent Transformers versions rewrite `image_url` entries to `image` before the chat template runs, so the video-detection branch in HCX's template never triggers and video inputs are silently dropped. As a workaround, use `processor.tokenizer.apply_chat_template` to render the prompt text, then pass the video path separately to `processor(...)`. See [this review comment](https://github.com/huggingface/transformers/pull/44314#discussion_r3008382827) for details.

- The model supports chain-of-thought reasoning. By default, the generation prompt prepends an empty `<think>\n\n</think>` block. To generate an explicit reasoning trace inside `<think>...</think>` tags, pass `thinking=True` to `apply_chat_template` (image/text inputs only):

    ```python
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        thinking=True,
    ).to(model.device)
    ```

- The model supports multi-turn conversations with mixed media. Images and videos can appear across multiple turns. For turns containing video, use the `processor.tokenizer.apply_chat_template` workaround described above.

    ```python
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}},
                {"type": "text", "text": "What do you see in this image?"},
            ],
        },
        {
            "role": "assistant",
            "content": "I see a cat sitting on a couch.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}},
                {"type": "text", "text": "How does this compare to the first image?"},
            ],
        },
    ]
    ```

- The model supports function/tool calling. Pass tools using the `tools` parameter in `apply_chat_template`:

    ```python
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What is the weather in Seoul?"}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    ```

- For text-only inference with the HyperCLOVAX language model backbone, use [`HyperCLOVAXForCausalLM`]:

    ```python
    import torch
    from transformers import HyperCLOVAXForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-32B")
    model = HyperCLOVAXForCausalLM.from_pretrained(
        "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    inputs = tokenizer("HyperCLOVAX is", return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    ```

## HyperCLOVAXConfig

[[autodoc]] HyperCLOVAXConfig

## HCXVisionV2Config

[[autodoc]] HCXVisionV2Config

## HCXVisionV2Processor

[[autodoc]] HCXVisionV2Processor
    - __call__

## HyperCLOVAXModel

[[autodoc]] HyperCLOVAXModel
    - forward

## HyperCLOVAXForCausalLM

[[autodoc]] HyperCLOVAXForCausalLM
    - forward

## HyperCLOVAXForSequenceClassification

[[autodoc]] HyperCLOVAXForSequenceClassification
    - forward

## HCXVisionV2Model

[[autodoc]] HCXVisionV2Model
    - forward
    - get_image_features
    - get_video_features

## HCXVisionV2ForConditionalGeneration

[[autodoc]] HCXVisionV2ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

## HCXVisionV2ForSequenceClassification

[[autodoc]] HCXVisionV2ForSequenceClassification
    - forward
