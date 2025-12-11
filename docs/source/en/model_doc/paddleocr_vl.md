<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025.10.16 and added to Hugging Face Transformers on 2025.12.10*

# PaddleOCR-VL

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**Huggingface Hub**: [PaddleOCR-VL](https://huggingface.co/collections/PaddlePaddle/paddleocr-vl) | **Github Repo**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

**Official Website**: [Baidu AI Studio](https://aistudio.baidu.com/paddleocr) | **arXiv**: [Technical Report](https://arxiv.org/pdf/2510.14528)

**PaddleOCR-VL** is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

<div align="center">
<img src="https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/allmetric.png" width="800"/>
</div>

### **Core Features**

1. **Compact yet Powerful VLM Architecture:** We present a novel vision-language model that is specifically designed for resource-efficient inference, achieving outstanding performance in element recognition. By integrating a NaViT-style dynamic high-resolution visual encoder with the lightweight ERNIE-4.5-0.3B language model, we significantly enhance the model’s recognition capabilities and decoding efficiency. This integration maintains high accuracy while reducing computational demands, making it well-suited for efficient and practical document processing applications.

2. **SOTA Performance on Document Parsing:** PaddleOCR-VL achieves state-of-the-art performance in both page-level document parsing and element-level recognition. It significantly outperforms existing pipeline-based solutions and exhibiting strong competitiveness against leading vision-language models (VLMs) in document parsing. Moreover, it excels in recognizing complex document elements, such as text, tables, formulas, and charts, making it suitable for a wide range of challenging content types, including handwritten text and historical documents. This makes it highly versatile and suitable for a wide range of document types and scenarios.

3. **Multilingual Support:** PaddleOCR-VL Supports 109 languages, covering major global languages, including but not limited to Chinese, English, Japanese, Latin, and Korean, as well as languages with different scripts and structures, such as Russian (Cyrillic script), Arabic, Hindi (Devanagari script), and Thai. This broad language coverage substantially enhances the applicability of our system to multilingual and globalized document processing scenarios.

### **Model Architecture** 

<div align="center">
<img src="https://huggingface.co/datasets/PaddlePaddle/PaddleOCR-VL_demo/resolve/main/imgs/paddleocrvl.png" width="800"/>
</div>

## Usage

### Usage tips

> [!IMPORTANT]
> We currently recommend using the [PaddleOCR official method for inference](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html), as it is faster and supports page-level document parsing. 
> The example code below only supports element-level recognition.

We have four types of element-level recognition:

- Text recognition, indicated by the prompt `OCR:`.
- Formula recognition, indicated by the prompt `Formula Recognition:`.
- Table recognition, indicated by the prompt `Table Recognition:`.
- Chart recognition, indicated by the prompt `Chart Recognition:`.

The following examples are all based on text recognition, with the prompt `OCR:`.

### Single input inference

The example below demonstrates how to generate text with PaddleOCRVL using [`Pipeline`] or the [`AutoModel`].

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg"},
            {"type": "text", "text": "OCR:"},
        ]
    }
]
result = pipe(text=messages)
print(result[0]["generated_text"])
```

</hfoption>

<hfoption id="AutoModel">

```py
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg"},
            {"type": "text", "text": "OCR:"},
        ]
    }
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
print(result)
```

</hfoption>
</hfoptions>

### Batched inference

PaddleOCRVL also supports batched inference. We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Here is how you can do it with PaddleOCRVL using [`Pipeline`] or the [`AutoModel`]:

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg"},
            {"type": "text", "text": "OCR:"},
        ]
    }
]
result = pipe(text=[messages, messages])
print(result[0][0]["generated_text"])
print(result[1][0]["generated_text"])
```

</hfoption>

<hfoption id="AutoModel">

```py
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype="bfloat16")
processor = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/ocr_demo2.jpg"},
            {"type": "text", "text": "OCR:"},
        ]
    }
]
batch_messages = [messages, messages]
inputs = processor.apply_chat_template(
	batch_messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    padding=True,
    padding_side='left',
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
result = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(result)
```

</hfoption>
</hfoptions>

### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization, please refer to the [FlashAttention](https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention).

For example:

```shell
pip install flash-attn --no-build-isolation
```

```python
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype="bfloat16", attn_implementation="flash_attention_2")
```

## PaddleOCRVLForConditionalGeneration

[[autodoc]] PaddleOCRVLForConditionalGeneration
    - forward

## PaddleOCRVLConfig

[[autodoc]] PaddleOCRVLConfig

## PaddleOCRVisionConfig

[[autodoc]] PaddleOCRVisionConfig

## PaddleOCRTextConfig

[[autodoc]] PaddleOCRTextConfig

## PaddleOCRTextModel

[[autodoc]] PaddleOCRTextModel

## PaddleOCRVisionModel

[[autodoc]] PaddleOCRVisionModel

## PaddleOCRVLImageProcessor

[[autodoc]] PaddleOCRVLImageProcessor

## PaddleOCRVLImageProcessorFast

[[autodoc]] PaddleOCRVLImageProcessorFast

## PaddleOCRVLModel

[[autodoc]] PaddleOCRVLModel

## PaddleOCRVLProcessor

[[autodoc]] PaddleOCRVLProcessor

## PaddleOCRVisionTransformer

[[autodoc]] PaddleOCRVisionTransformer
