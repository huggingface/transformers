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
*This model was published in HF papers on 2024-08-03 and contributed to Hugging Face Transformers on 2026-11-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MiniCPM-V 2.6

[MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [MiniCPM-V 4.0](https://huggingface.co/openbmb/MiniCPM-V-4) are multimodal models from [OpenBMB](https://github.com/OpenBMB) ([paper](https://huggingface.co/papers/2408.01800)) built like [MiniCPM-Llama3-V 2.5](minicpmv2_5), with a resampler of 64 learned queries. The vision encoder's position embedding follows each slice's 2D patch grid, and every slice has its own `<slice>...</slice>` placeholder. MiniCPM-V 2.6 uses a [Qwen2](qwen2) 7B language model, MiniCPM-V 4.0 uses MiniCPM4-3B, a [Llama](llama)-architecture model with LongRoPE. Images are prepared by [`MiniCPMV2_5ImageProcessor`].

The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

## Usage example

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4"
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
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

## Notes

- In a prompt, each image is marked with `<unk>`. The processor expands it to `<image>` + 64 × `<unk>` + `</image>` for the overview image, followed by each slice as `<slice>` + 64 × `<unk>` + `</slice>`, one line per row of slices.
- The chat template puts `<image_id>N</image_id>` before each image, N being the image's index in the conversation, as the original processor does. Add it yourself when you write prompts without the chat template.
- The vision encoder's position ids bucket each patch by `index * (1 / size)` in its slice's patch grid, as Idefics2 does. The original code computes the same buckets with a CPU `torch.arange`, whose rounding depends on the CPU, so on some slice sizes (49 and 50 patches on ARM) a few patches land one bucket off from the original.

## MiniCPMV2_6Config

[[autodoc]] MiniCPMV2_6Config

## MiniCPMV2_6VisionConfig

[[autodoc]] MiniCPMV2_6VisionConfig

## MiniCPMV2_6VisionTransformer

[[autodoc]] MiniCPMV2_6VisionTransformer
    - forward

## MiniCPMV2_6Model

[[autodoc]] MiniCPMV2_6Model
    - forward
    - get_image_features

## MiniCPMV2_6ForConditionalGeneration

[[autodoc]] MiniCPMV2_6ForConditionalGeneration
    - forward
    - get_image_features

## MiniCPMV2_6Processor

[[autodoc]] MiniCPMV2_6Processor
    - __call__
