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

*This model was released on 2025-05-06 and added to Hugging Face Transformers on 2025-12-02.*

# FastVLM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

FastVLM is an open-source vision-language model featuring a novel hybrid vision encoder, FastViTHD. Leveraging reparameterizable convolutional layers, scaled input resolution, and a reduced number of visual tokens, FastVLM delivers high accuracy with exceptional efficiency. Its optimized architecture enables deployment even on edge devices, achieving ultra-low TTFT (time to first token) without sacrificing performance.

The model was proposed in [FastVLM: Efficient Vision Encoding for Vision Language Models](https://huggingface.co/papers/2412.13303) by Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel and Hadi Pouransari.

The abstract from the paper is the following:

*Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs become inefficient at high resolutions due to the large number of tokens and high encoding latency. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and  minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, token count, and LLM size, we introduce FastVLM—a model that achieves an optimized trade-off between resolution, latency, and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image resolution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2× improvement in time-to-first-token (TTFT) while maintaining similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the highest resolution (1152×1152), FastVLM achieves better performance on key benchmarks like SeedBench, MMMU and DocVQA, using the same 0.5B LLM, but with 85× faster TTFT and a vision encoder that is 3.4× smaller.*

This model was contributed by [Kamila](https://github.com/kamila-chay).
The original code can be found [here](https://github.com/apple/ml-fastvlm).

## Usage tips

- We advise users to use `padding_side="left"` when computing batched generation as it leads to more accurate results. Simply make sure to call `processor.tokenizer.padding_side = "left"` before generating.

- Note the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results.

**Important:**

Hugging Face models use SDPA by default; however, this model’s visual backbone supports only eager attention, so it automatically falls back to `"eager"`.

If you want to use a different attention implementation in the language decoder, make sure to set it explicitly, for example:

`model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", attn_implementation={"text_config": "flash_attention_2"})`

Setting it for the entire model, e.g.

`model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", attn_implementation="flash_attention_2")`

will result in an error.

### Formatting Prompts with Chat Templates

Each **checkpoint** is trained with a specific prompt format, depending on the underlying large language model backbone. To ensure correct formatting, use the processor’s `apply_chat_template` method.

**Important:**

- You must construct a conversation history — passing a plain string won't work.
- Each message should be a dictionary with `"role"` and `"content"` keys.
- The `"content"` should be a list of dictionaries for different modalities like `"text"` and `"image"`.

## Usage examples

### Single input inference

```python
import torch
from transformers import AutoProcessor, FastVlmForConditionalGeneration

# Load the model in half-precision
model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device, torch.bfloat16)

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```

### Batched inference

FastVLM also supports batched inference. Here is how you can do it:

```python
import torch
from transformers import AutoProcessor, FastVlmForConditionalGeneration

# Load the model in half-precision
model = FastVlmForConditionalGeneration.from_pretrained("KamilaMila/FastVLM-0.5B", dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained("KamilaMila/FastVLM-0.5B")


# Prepare a batch of two prompts
conversation_1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

conversation_2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    [conversation_1, conversation_2],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    padding=True,
    return_tensors="pt"
).to(model.device, torch.bfloat16)


# Generate
generate_ids = model.generate(**inputs, max_new_tokens=30)
processor.batch_decode(generate_ids, skip_special_tokens=True)
```

## Note regarding reproducing original implementation

In order to match the logits of the [original implementation](https://github.com/apple/ml-fastvlm), one needs to use float32. In half precision the logit difference is higher due to tiny differences in how some ops are implemented in timm.

### Using Flash Attention 2

Flash Attention 2 is an even faster, optimized version of the previous optimization, please refer to the [Flash Attention 2 section of performance docs](https://huggingface.co/docs/transformers/perf_infer_gpu_one).

## FastVlmConfig

[[autodoc]] FastVlmConfig

## FastVlmModel

[[autodoc]] FastVlmModel

## FastVlmForConditionalGeneration

[[autodoc]] FastVlmForConditionalGeneration
    - forward
    - get_image_features
