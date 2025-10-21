
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
*This model was released on 2025-03-25 and added to Hugging Face Transformers on 2025-03-12 and contributed by [RyanMullins](https://huggingface.co/RyanMullins), [RaushanTurganbay](https://huggingface.co/RaushanTurganbay), [ArthurZ](https://huggingface.co/ArthurZ), and [pcuenq](https://huggingface.co/pcuenq).*

# Gemma3

[Gemma 3](https://huggingface.co/papers/2503.19786) is a multimodal extension of the Gemma model family, spanning 1 – 27 billion parameters and adding vision understanding, broader multilingual capability, and support for long 128K-token contexts. Its architecture reduces KV-cache memory growth by increasing the proportion of local attention layers and shortening their attention spans. The models are trained via distillation and use a new post-training recipe that greatly boosts math, chat, instruction-following, and multilingual performance. As a result, Gemma3-4B-IT matches the performance of the much larger Gemma2-27B-IT, while Gemma3-27B-IT rivals Gemini-1.5-Pro across benchmarks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
Copied
import torch
from transformers import pipeline

pipeline = pipeline(task="image-text-to-text", model="google/gemma-3-4b-pt", dtype="auto")
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

</hfoption>
<hfoption id="Gemma3ForConditionalGeneration">

```py
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it", dtype="auto")
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", padding_side="left")

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

- Use [`Gemma3ForConditionalGeneration`] for image-and-text and image-only inputs.
- Add `<start_of_image>` tokens to text wherever images should be inserted.
- The processor includes [`apply_chat_template`] to convert chat messages to model inputs.
- By default, images aren't cropped and only the base image forwards to the model. High-resolution images or non-square aspect ratios can cause artifacts because the vision encoder uses a fixed 896×896 resolution. Set `do_pan_and_scan=True` to crop images into multiple smaller patches and concatenate them with the base image embedding. This prevents artifacts and improves inference performance. Disable pan and scan for faster inference.
- For Gemma-3 1B checkpoint trained in text-only mode, use [`AutoModelForCausalLM`] instead.

## Gemma3ImageProcessor

[[autodoc]] Gemma3ImageProcessor

## Gemma3ImageProcessorFast

[[autodoc]] Gemma3ImageProcessorFast

## Gemma3Processor

[[autodoc]] Gemma3Processor

## Gemma3TextConfig

[[autodoc]] Gemma3TextConfig

## Gemma3Config

[[autodoc]] Gemma3Config

## Gemma3TextModel

[[autodoc]] Gemma3TextModel
    - forward

## Gemma3ForCausalLM

[[autodoc]] Gemma3ForCausalLM
    - forward

## Gemma3ForConditionalGeneration

[[autodoc]] Gemma3ForConditionalGeneration
    - forward

## Gemma3Model

[[autodoc]] Gemma3Model
    - forward

## Gemma3TextForSequenceClassification

[[autodoc]] Gemma3TextForSequenceClassification
    - forward

## Gemma3ForSequenceClassification

[[autodoc]] Gemma3ForSequenceClassification
    - forward
