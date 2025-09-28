
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
*This model was released on 2025-09-18 and added to Hugging Face Transformers on 2024-09-25.*

# LLaVA-OneVision

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

**LLaVA-OneVision1.5** introduces a novel family of **fully open-source** Large Multimodal Models (LMMs) that achieves **state-of-the-art performance**  with substantially **lower cost** through training on **native resolution** images.

- **Superior Performance**
A family of fully open-source large multimodal models demonstrating 
    - Superior performance across multiple multimodal benchmarks
    - outperforming **Qwen2.5-VL** in most evaluation tasks.

- **High-Quality Data at Scale**
Meticulously curated **pre-training and SFT data** with rigorous filtering and quality control, achieving **superior data efficiency** with only **64B tokens**.
    - Concept-balanced, highly diverse, high-quality caption data
    - Comprehensive instruction fine-tuning data covering a wide range of tasks

- **Ultra-Efficient Training Framework** Complete end-to-end training framework designed for maximum efficiency:
    - $16000 total budget for full model training on A100 GPUs  ($0.6 per GPU/Hour)
    - 45% HFU efficiency in 8k context length
    - Built on **MegatronLM** with support for **MoE**, **FP8**, and **long sequence parallelization**
    - Optimized codebase for cost-effective scaling


- **Fully Open Framework** for community access and reproducibility:
    - High-quality pre-training & SFT data
    - Complete training framework & code
    - Training recipes & configurations
    - Comprehensive training logs & metrics


## Quick Start with HuggingFace

```python
from transformers import AutoProcessor, LlavaOnevision1_5ForConditionalGeneration
from qwen_vl_utils import process_vision_info
model_path = "lmms-lab/LLaVA-One-Vision-1.5-8B-Instruct"

# default: Load the model on the available device(s)
model = LlavaOnevision1_5ForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processer
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```


## LlavaOnevision1_5Config

[[autodoc]] LlavaOnevision1_5Config

## RicePretrainedModel

[[autodoc]] RicePretrainedModel
    - forward

## LlavaOnevisionModel1_5Model

[[autodoc]] LlavaOnevisionModel1_5Model
    - forward

## LlavaOnevision1_5ForConditionalGeneration

[[autodoc]] LlavaOnevision1_5ForConditionalGeneration
    - forward