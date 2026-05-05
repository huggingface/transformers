<!--Copyright 2026 IBM and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-03-27 and added to Hugging Face Transformers on 2026-05-05.*

<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
  </div>
</div>

# Granite4Vision

[Granite Vision 4.1](https://huggingface.co/ibm-granite/granite-vision-4.1-4b) is a vision-language model from IBM Research designed for enterprise-grade document data extraction. It specializes in chart extraction (Chart2CSV, Chart2Summary, Chart2Code), table extraction (JSON, HTML, OTSL), and semantic key-value pair extraction.

The model builds on [LLaVA-NeXT](llava_next) with several architectural innovations:

1. **SigLIP2 Vision Encoder** ([`google/siglip2-so400m-patch16-384`](https://huggingface.co/google/siglip2-so400m-patch16-384)): images are tiled into 384x384 patches.
2. **Window Q-Former Projectors**: compress visual features 4x using windowed cross-attention over 4x4 patch windows into 2x2 tokens.
3. **DeepStack Feature Injection** with 8 vision-to-LLM injection points:
   - *LayerDeepstack*: features from 4 vision encoder depths are projected into different early LLM layers.
   - *SpatialDeepstack*: deepest vision features are split into 4 spatial groups and injected at later LLM layers.
4. **Language Model**: [Granite 4.1](https://huggingface.co/ibm-granite/granite-4.1-4b-base) (4B params) with LoRA adapters (rank 256) across all self-attention and MLP layers.

The model is delivered as a LoRA adapter on top of the base LLM, enabling single deployments to support both multimodal and text-only workloads. Total parameter count is ~4B.

> [!TIP]
> This model was contributed by the [IBM Granite Vision Team](https://github.com/ibm-granite).

## Usage Tips

- Set `padding_side="left"` during batched generation for more accurate results.

```py
processor.tokenizer.padding_side = "left"
```

- The model supports specialized task tags for document extraction: `<chart2csv>`, `<chart2summary>`, `<chart2code>`, `<tables_html>`, `<tables_otsl>`, `<tables_json>`. Pass these as the text prompt along with a document image.

- For key-value pair extraction, provide a JSON schema describing the fields to extract. The model returns structured JSON matching the schema.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">

<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(
    task="image-text-to-text",
    model="ibm-granite/granite-vision-4.1-4b",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
pipe(text=messages, max_new_tokens=100, return_full_text=False)
```

</hfoption>

<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "ibm-granite/granite-vision-4.1-4b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>

</hfoptions>

Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the [Quantization](../quantization/overview) overview for more available quantization backends.

The example below uses [bitsandbytes](../quantization/bitsandbytes) to only quantize the weights to int4.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

model_id = "ibm-granite/granite-vision-4.1-4b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, quantization_config=quant_config, device_map="auto"
)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
```

## Granite4VisionConfig

[[autodoc]] Granite4VisionConfig

## Granite4VisionTextConfig

[[autodoc]] Granite4VisionTextConfig

## Granite4VisionProcessor

[[autodoc]] Granite4VisionProcessor
    - __call__

## Granite4VisionModel

[[autodoc]] Granite4VisionModel

## Granite4VisionTextModel

[[autodoc]] Granite4VisionTextModel

## Granite4VisionForConditionalGeneration

[[autodoc]] Granite4VisionForConditionalGeneration
    - forward
    - get_image_features
