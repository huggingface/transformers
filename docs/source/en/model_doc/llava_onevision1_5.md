
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

# LLaVA-OneVision 1.5

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The LLaVA-OneVision 1.5 model was proposed in [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://huggingface.co/papers/2509.23661) by <Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, Ziyong Feng, Ziwei Liu, Bo Li, Jiankang Deng

**LLaVA-OneVision 1.5** introduces a novel family of **fully open-source** Large Multimodal Models (LMMs) that achieves **state-of-the-art performance**  with substantially **lower cost** through training on **native resolution** images. 

The abstract from the paper is the following:

*We present LLaVA-OneVision-1.5, a novel family of Large Multimodal Models (LMMs) that achieve state-of-the-art performance with significantly reduced computational and financial costs. Different from the existing works, LLaVA-OneVision-1.5 provides an open, efficient, and reproducible framework for building high-quality vision-language models entirely from scratch. The LLaVA-OneVision-1.5 release comprises three primary components: (1) Large-Scale Curated Datasets: We construct an 85M concept-balanced pretraining dataset LLaVA-OneVision-1.5-Mid-Traning and a meticulously curated 22M instruction dataset LLaVA-OneVision-1.5-Instruct. (2) Efficient Training Framework: We develop a complete end-to-end efficient training framework leveraging an offline parallel data packing strategy to facilitate the training of LLaVA-OneVision-1.5 within a $16,000 budget. (3) State-of-the-art Performance: Experimental results demonstrate that LLaVA-OneVision-1.5 yields exceptionally competitive performance across a broad range of downstream tasks. Specifically, LLaVA-OneVision-1.5-8B outperforms Qwen2.5-VL-7B on 18 of 27 benchmarks, and LLaVA-OneVision-1.5-4B surpasses Qwen2.5-VL-3B on all 27 benchmarks. We anticipate releasing LLaVA-OneVision-1.5-RL shortly and encourage the community to await further updates.*


## Quick Start with HuggingFace

```python
from transformers import AutoProcessor, LlavaOnevision1_5ForConditionalGeneration

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
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

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


## LlavaOnevision1_5Model

[[autodoc]] LlavaOnevision1_5Model
    - forward

## LlavaOnevision1_5ForConditionalGeneration

[[autodoc]] LlavaOnevision1_5ForConditionalGeneration
    - forward