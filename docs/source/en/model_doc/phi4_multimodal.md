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
*This model was released on 2025-03-03 and added to Hugging Face Transformers on 2025-03-25 and contributed by [cyrilvallez](https://huggingface.co/cyrilvallez).*

## Phi4 Multimodal

[Phi4 Multimodal](https://huggingface.co/papers/2503.01743) extends Phi-4-Mini to handle text, vision, and speech/audio inputs, using LoRA adapters and modality-specific routers to support multiple inference modes without interference. Both models outperform larger counterparts in reasoning, vision-language, and speech-language tasks despite their compact sizes.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
import torch
from transformers import pipeline

pipeline = pipeline("text-generation", model="microsoft/Phi-4-multimodal-instruct", dtype="auto")
pipeline("Plants create energy through a process known as ")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/Phi-4-multimodal-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-multimodal-instruct", dtype="auto")

model.load_adapter(model_path, adapter_name="vision", device_map=device, adapter_kwargs={"subfolder": "vision-lora"})

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

model.set_adapter("vision")
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
```

</hfoption>
</hfoptions>

## Phi4MultimodalFeatureExtractor

[[autodoc]] Phi4MultimodalFeatureExtractor

## Phi4MultimodalImageProcessorFast

[[autodoc]] Phi4MultimodalImageProcessorFast

## Phi4MultimodalProcessor

[[autodoc]] Phi4MultimodalProcessor

## Phi4MultimodalAudioConfig

[[autodoc]] Phi4MultimodalAudioConfig

## Phi4MultimodalVisionConfig

[[autodoc]] Phi4MultimodalVisionConfig

## Phi4MultimodalConfig

[[autodoc]] Phi4MultimodalConfig

## Phi4MultimodalAudioModel

[[autodoc]] Phi4MultimodalAudioModel

## Phi4MultimodalVisionModel

[[autodoc]] Phi4MultimodalVisionModel

## Phi4MultimodalModel

[[autodoc]] Phi4MultimodalModel
    - forward

## Phi4MultimodalForCausalLM

[[autodoc]] Phi4MultimodalForCausalLM
    - forward
