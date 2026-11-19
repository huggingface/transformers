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
*This model was published in HF papers on 2024-08-03 and contributed to Hugging Face Transformers on 2026-11-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MiniCPM-Llama3-V 2.5

[MiniCPM-Llama3-V 2.5](https://huggingface.co/papers/2408.01800) is an 8B multimodal model from [OpenBMB](https://github.com/OpenBMB). An image is cut into an overview plus up to 9 slices. Each slice goes through an [Idefics2](idefics2) (SigLIP-based) vision encoder, and a cross-attention resampler with 96 learned queries turns it into 96 tokens for a [Llama 3](llama3) 8B language model.

The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

## Usage example

```python
from transformers import AutoProcessor, MiniCPMV2_5ForConditionalGeneration

model_id = "openbmb/MiniCPM-Llama3-V-2_5"
model = MiniCPMV2_5ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
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
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

## Notes

- In a prompt, each image is marked with `<unk>`. The processor expands it to `<image>` + 96 × `<unk>` + `</image>` for the overview image, followed by the slices row by row inside `<slice>...</slice>`.
- The original remote code never masked the zero padding added when slices of different sizes are batched together, so a slice's features depended on the other images in the batch. This implementation masks the padding, so each slice is encoded independently. Outputs therefore differ slightly from the original code, whose results it reproduces token-for-token when that behavior is replicated.

## MiniCPMV2_5Config

[[autodoc]] MiniCPMV2_5Config

## MiniCPMV2_5VisionConfig

[[autodoc]] MiniCPMV2_5VisionConfig

## MiniCPMV2_5Model

[[autodoc]] MiniCPMV2_5Model
    - forward
    - get_image_features

## MiniCPMV2_5ForConditionalGeneration

[[autodoc]] MiniCPMV2_5ForConditionalGeneration
    - forward
    - get_image_features

## MiniCPMV2_5Processor

[[autodoc]] MiniCPMV2_5Processor
    - __call__

## MiniCPMV2_5ImageProcessor

[[autodoc]] MiniCPMV2_5ImageProcessor
    - preprocess

## MiniCPMV2_5ImageProcessorPil

[[autodoc]] MiniCPMV2_5ImageProcessorPil
    - preprocess
