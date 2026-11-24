<!--Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-09-16 and contributed to Hugging Face Transformers on 2026-11-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MiniCPM-V 4.5

[MiniCPM-V 4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) is a multimodal model from [OpenBMB](https://github.com/OpenBMB). For images it is [MiniCPM-V 2.6](minicpmv2_6) with a [Qwen3](qwen3) 8B language model, and it uses [`MiniCPMV2_6Processor`]. Video input, which the original resamples across frames, isn't supported yet.

The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

## Usage example

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4_5"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

## MiniCPMV4_5Config

[[autodoc]] MiniCPMV4_5Config

## MiniCPMV4_5VisionConfig

[[autodoc]] MiniCPMV4_5VisionConfig

## MiniCPMV4_5VisionTransformer

[[autodoc]] MiniCPMV4_5VisionTransformer
    - forward

## MiniCPMV4_5Model

[[autodoc]] MiniCPMV4_5Model
    - forward
    - get_image_features

## MiniCPMV4_5ForConditionalGeneration

[[autodoc]] MiniCPMV4_5ForConditionalGeneration
    - forward
    - get_image_features
