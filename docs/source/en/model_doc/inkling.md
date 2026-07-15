<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-15.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Inkling

[Inkling](https://huggingface.co/thinkingmachines/Inkling) is a general-purpose multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text, image, and audio inputs and generates text. It is a 66-layer decoder-only transformer with a sparse mixture-of-experts (MoE) feed-forward backbone — each token is routed to 6 of 256 experts alongside 2 shared experts that are always active — for 975B total parameters with 41B active per token. Image and audio inputs are projected into the language model's embedding space and interleaved with text tokens, so a single checkpoint reasons jointly over all three modalities.

You can find the official checkpoints under the [Thinking Machines Lab](https://huggingface.co/thinkingmachines) organization.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

model_id = "thinkingmachines/Inkling-NVFP4"
pipe = pipeline("any-to-any", model=model_id)

image_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/pills.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text",
                "text": "Do components in this supplement interact with each other?",
            },
        ],
    },
]
output = pipe(
    messages,
    max_new_tokens=2000,
    return_full_text=False,
    reasoning_effort="medium",
)
output[0]["generated_text"]
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-NVFP4"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You should only answer with a number."},
    {"role": "user", "content": "What is 17 * 23?"},
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=2000)
generated_tokens = output[0][inputs["input_ids"].shape[1] :]
print(processor.decode(generated_tokens, skip_special_tokens=False))
```

</hfoption>
</hfoptions>

## Notes

- Text and image inference:

```py
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    device_map="auto",
)

image_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/pills.jpg"
)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {
                "type": "text",
                "text": "Do any of the components in this supplement interact?",
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    reasoning_effort="medium",
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=2000)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```

- Text with audio inference:

```py
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    device_map="auto",
)

audio_url = (
    "https://huggingface.co/datasets/merve/vl-test-suite/"
    "resolve/main/example_audio.mp3"
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe the following speech to text."},
            {
                "type": "audio",
                "audio": audio_url,
            },
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

processor.parse_response(response)
```

## InklingAudioConfig

[[autodoc]] InklingAudioConfig

## InklingConfig

[[autodoc]] InklingConfig

## InklingTextConfig

[[autodoc]] InklingTextConfig

## InklingVisionConfig

[[autodoc]] InklingVisionConfig

## InklingAudioModel

[[autodoc]] InklingAudioModel
    - forward

## InklingForCausalLM

[[autodoc]] InklingForCausalLM

## InklingForConditionalGeneration

[[autodoc]] InklingForConditionalGeneration

## InklingModel

[[autodoc]] InklingModel
    - forward

## InklingPreTrainedModel

[[autodoc]] InklingPreTrainedModel
    - forward

## InklingTextModel

[[autodoc]] InklingTextModel
    - forward

## InklingVisionModel

[[autodoc]] InklingVisionModel
    - forward

## InklingImageProcessor

[[autodoc]] InklingImageProcessor

## InklingFeatureExtractor

[[autodoc]] InklingFeatureExtractor

## InklingProcessor

[[autodoc]] InklingProcessor
