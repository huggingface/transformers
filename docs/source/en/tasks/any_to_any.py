<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

## Pipeline

The fastest way to get started is to use the [`Pipeline`] API. Specify the `"multimodal-generation"` task and the model you want to use.

```python
from transformers import pipeline
pipe = pipeline("multimodal-generation", model="google/gemma-3n-E4B-it")
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

If you prefer, you can also load images as input like so:

```python
messages = [
     {
         "role": "user",
         "content": [
             {
                 "type": "image",
                 "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
             },
             {"type": "text", "text": "What do you hear in this audio?"},
         ],
     },
]
outputs = pipe(text=messages, images=images, max_new_tokens=50, return_full_text=False)
outputs[0]["generated_text"]
```

Multimodal Pipeline also supports generating audio or images with any-to-any models. For that you need to set `generation_mode` parameter and set video sampling to the desired FPS as follows:

```python
import soundfile as sf
pipe = pipeline("multimodal-generation", model="Qwen/Qwen2.5-Omni-3B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "path": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Cooking_cake.mp4"},
            {"type": "text", "text": "Describe this video."},
        ],
    },
]
output = pipe(text=messages, fps=2, use_audio_in_video=True, max_new_tokens=20, generation_mmode="audio")
sf.write("generated_audio.wav", out[0]["generated_audio"])
```