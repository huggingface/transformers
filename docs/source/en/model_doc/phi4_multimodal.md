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

# Phi4 Multimodal

## Overview

Phi4 Multimodal is a lightweight open multimodal foundation model that leverages the language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models. The model processes text, image, and audio inputs, generating text outputs, and comes with 128K token context length. The model underwent an enhancement process, incorporating both supervised fine-tuning, direct preference optimization and RLHF (Reinforcement Learning from Human Feedback) to support precise instruction adherence and safety measures. The languages that each modal supports are the following:

- Text: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian
- Vision: English
- Audio: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese

This model was contributed by [Cyril Vallez](https://huggingface.co/cyrilvallez). The most recent code can be
found [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py).


## Usage tips

`Phi4-multimodal-instruct` can be found on the [Huggingface Hub](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

In the following, we demonstrate how to use it for inference depending on the input modalities (text, image, audio).

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# Define model path
model_path = "microsoft/Phi-4-multimodal-instruct"
device = "cuda:0"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device,  torch_dtype=torch.float16)

# Optional: load the adapters (note that without them, the base model will very likely not work well)
model.load_adapter(model_path, adapter_name="speech", device_map=device, adapter_kwargs={"subfolder": 'speech-lora'})
model.load_adapter(model_path, adapter_name="vision", device_map=device, adapter_kwargs={"subfolder": 'vision-lora'})

# Part : Image Processing
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

model.set_adapter("vision") # if loaded, activate the vision adapter
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

# Generate response
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


# Part 2: Audio Processing
model.set_adapter("speech") # if loaded, activate the speech adapter
audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
            {"type": "text", "text": "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the origina transcript and the translation."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

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
