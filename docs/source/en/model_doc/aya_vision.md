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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# AyaVision

[AyaVision](https://huggingface.co/papers/2505.08751) is a family of open-weight multimodal vision-language models from Cohere Labs. Each model integrates a Command R7B–based multilingual transformer that has been post-trained with a synthetic multimodal instruction pipeline, together with a SigLIP2-patched vision encoder connected via lightweight adapter layers. The system processes images as tiled 364 × 364 patches plus a low-resolution thumbnail and supports up to 16 K tokens of text. It performs OCR, image captioning, visual reasoning, multilingual question answering in 23 languages, and code-comment generation without compromising text-only tasks.

You can find all the original AyaVision checkpoints under the [AyaVision](https://huggingface.co/collections/CohereLabs/cohere-labs-aya-vision-67c4ccd395ca064308ee1484) collection.

> [!TIP]
> Click on the AyaVision models in the right sidebar for more examples of how to apply AyaVision to different image-to-text tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline(model="CohereLabs/aya-vision-8b", task="image-text-to-text", device_map="auto")

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://media.istockphoto.com/id/458012057/photo/istanbul-turkey.jpg?s=612x612&w=0&k=20&c=qogAOVvkpfUyqLUMr_XJQyq-HkACXyYUSZbKhBlPrxo="},
        {"type": "text", "text": "Bu resimde hangi anıt gösterilmektedir?"},
    ]},
    ]
outputs = pipe(text=messages, max_new_tokens=300, return_full_text=False)

print(outputs)

```

</hfoption>
<hfoption id="AutoModel">

```python
# pip install 'git+https://github.com/huggingface/transformers.git@v4.49.0-AyaVision'
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

model_id = "CohereLabs/aya-vision-8b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

# Format message with the aya-vision chat template
messages = [
    {"role": "user",
     "content": [
       {"type": "image", "url": "https://pbs.twimg.com/media/Fx7YvfQWYAIp6rZ?format=jpg&name=medium"},
        {"type": "text", "text": "चित्र में लिखा पाठ क्या कहता है?"},
    ]},
    ]

inputs = processor.apply_chat_template(
    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

gen_tokens = model.generate(
    **inputs,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
)

print(processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))

```

</hfoption>

</hfoptions>

Quantization reduces the memory footprint of large models by representing weights at lower precision. Refer to the [Quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) overview for supported backends.

## Notes

- Use explicit language tokens (e.g. `<en>`, `<fr>`, `<hi>`) in prompts to reliably set the output language—otherwise it infers from input, which may be less precise.

- You can pass a list of messages to the `pipeline` or manually format them via `apply_chat_template`, allowing multiple images or prompts in a single forward pass.

- The model uses `use_cache=True` by default, so repeated `generate(...)` calls retain past key/value states and speed up decoding for long outputs.

## AyaVisionProcessor

[[autodoc]] AyaVisionProcessor

## AyaVisionConfig

[[autodoc]] AyaVisionConfig

## AyaVisionModel

[[autodoc]] AyaVisionModel

## AyaVisionForConditionalGeneration

[[autodoc]] AyaVisionForConditionalGeneration 
    - forward
