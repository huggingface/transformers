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

*This model was released on 2025-05-20 and added to Hugging Face Transformers on 2025-06-26.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Gemma3n

[Gemma3n](https://developers.googleblog.com/en/introducing-gemma-3n/) is a compact AI model built by Google DeepMind that uses Per-Layer Embeddings (PLE) to dramatically cut RAM usage, enabling 5B and 8B-parameter models to run with the memory footprint of 2B and 4B models (around 2–3 GB). It features MatFormer-based dynamic scaling, allowing developers to switch between a 4B model and an embedded 2B submodel for on-the-fly performance-quality trade-offs, and supports KVC sharing and activation quantization for faster, more efficient on-device inference. Gemma 3n also expands multimodal processing, handling audio, text, and images—including high-quality speech recognition, translation, and video understanding—and enhances multilingual performance across major languages. Designed for privacy-first local use, it runs fully offline with fast response times and low resource requirements, optimized for Android and Chrome deployment.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
Copied
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="google/gemma-3n-e4b", dtype="auto")
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

</hfoption>
<hfoption id="Gemma3nForConditionalGeneration">

```py
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3n-e4b", dtype="auto")
processor = AutoProcessor.from_pretrained("google/gemma-3n-e4b", padding_side="left")

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## Usage tips

- Use [`Gemma3nForConditionalGeneration`] for image-audio-and-text, image-and-text, image-and-audio, audio-and-text, image-only, and audio-only inputs.
- Add `<image_soft_token>` tokens to text wherever images should be inserted.
- Add `<audio_soft_token>` tokens to text wherever audio clips should be inserted.
- Gemma3n accepts at most one target audio clip per input. Multiple audio clips work in few-shot prompts.
- The processor includes [`apply_chat_template`] to convert chat messages to model inputs.

## Gemma3nAudioFeatureExtractor

[[autodoc]] Gemma3nAudioFeatureExtractor

## Gemma3nProcessor

[[autodoc]] Gemma3nProcessor

## Gemma3nTextConfig

[[autodoc]] Gemma3nTextConfig

## Gemma3nVisionConfig

[[autodoc]] Gemma3nVisionConfig

## Gemma3nAudioConfig

[[autodoc]] Gemma3nAudioConfig

## Gemma3nConfig

[[autodoc]] Gemma3nConfig

## Gemma3nTextModel

[[autodoc]] Gemma3nTextModel
    - forward

## Gemma3nModel

[[autodoc]] Gemma3nModel
    - forward

## Gemma3nForCausalLM

[[autodoc]] Gemma3nForCausalLM
    - forward

## Gemma3nForConditionalGeneration

[[autodoc]] Gemma3nForConditionalGeneration
    - forward

[altup]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html
[attention-mask-viz]: https://github.com/huggingface/transformers/blob/beb9b5b02246b9b7ee81ddf938f93f44cfeaad19/src/transformers/utils/attention_visualizer.py#L139
[gemma3n-collection]: https://huggingface.co/collections/google/gemma-3n
[laurel]: https://huggingface.co/papers/2411.07501
[matformer]: https://huggingface.co/papers/2310.07707
[spark-transformer]: https://huggingface.co/papers/2506.06644
[usm]: https://huggingface.co/papers/2303.01037

