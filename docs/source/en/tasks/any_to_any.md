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

# Multimodal Generation

[[open-in-colab]]

Multimodal (any-to-any) models are language models capable of processing diverse types of input data (e.g., text, images, audio, or video) and generating outputs in any of these modalities. Unlike traditional unimodal or fixed-modality models, they allow flexible combinations of input and output, enabling a single system to handle a wide range of tasks: from text-to-image generation to audio-to-text transcription, image captioning, video understanding, and so on. This task shares many similarities with image-text-to-text, but supports a wider range of input and output modalities.

In this guide, we provide a brief overview of any-to-any models and show how to use them with Transformers for inference. Unlike Vision LLMs, which are typically limited to vision-and-language tasks, omni-modal models can accept any combination of modalities (e.g., text, images, audio, video) as input, and generate outputs in different modalities, such as text or images.

Let’s begin by installing dependencies:

```bash
pip install -q transformers accelerate flash_attn
```

Let's initialize the model and the processor.

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM, infer_device
import torch

device = torch.device(infer_device())
model = AutoModelForMultimodalLM.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(device)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
```

These models typically include a [chat template](./chat_templating) to structure conversations across modalities. Inputs can mix images, text, audio, or other supported formats in a single turn. Outputs may also vary (e.g., text generation or audio generation), depending on the configuration.

Below is an example providing a "text + audio" input and requesting a text response.

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav"},
            {"type": "text", "text": "What do you hear in this audio?"},
        ]
    },
]
```

We will now call the processors' [`~ProcessorMixin.apply_chat_template`] method to preprocess its output along with the image inputs.

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)
```

We can now pass the preprocessed inputs to the model.

```python
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
```

## Pipeline

The fastest way to get started is to use the [`Pipeline`] API. Specify the `"any-to-any"` task and the model you want to use.

```python
from transformers import pipeline
pipe = pipeline("any-to-any", model="mistralai/Voxtral-Mini-3B-2507")
```

The example below uses chat templates to format the text inputs and uses audio modality as an multimodal data.

```python
messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "audio",
                 "url": "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
             },
             {"type": "text", "text": "What do you hear in this audio?"},
         ],
     },
 ]
```

Pass the chat template formatted text and image to [`Pipeline`] and set `return_full_text=False` to remove the input from the generated output.

```python
outputs = pipe(text=messages, max_new_tokens=20, return_full_text=False)
outputs[0]["generated_text"]
```

Any-to-any pipeline also supports generating audio or images with any-to-any models. For that you need to set `generation_mode` parameter. Do not forget to set video sampling to the desired FPS, otherwise the whole video will be loaded without sampling. Here is an example code:

```python
import soundfile as sf
pipe = pipeline("any-to-any", model="Qwen/Qwen2.5-Omni-3B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Cooking_cake.mp4"},
            {"type": "text", "text": "Describe this video."},
        ],
    },
]
output = pipe(text=messages, fps=1, load_audio_from_video=True, max_new_tokens=20, generation_mode="audio")
sf.write("generated_audio.wav", out[0]["generated_audio"])
```

