
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
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Gemma3n

## Overview

Gemma3n is a multimodal model with pretrained and instruction-tuned variants, available in E4B and E2B sizes. While
large portions of the language model architecture are shared with prior Gemma releases, there are many new additions in
this model, including [Alternating Updates][altup] (AltUp), [Learned Augmented Residual Layer][laurel] (LAuReL),
[MatFormer][matformer], Per-Layer Embeddings (PLE), [Activation Sparsity with Statistical Top-k][spark-transformer], and KV cache sharing. The language model uses
a similar attention pattern to [Gemma 3](./gemma3) with alternating 4 local sliding window self-attention layers for
every global self-attention layer with a maximum context length of 32k tokens. Gemma 3n introduces
[MobileNet v5][mobilenetv5] as the vision encoder, using a default resolution of 768x768 pixels, and adds a newly
trained audio encoder based on the [Universal Speech Model][usm] (USM) architecture.

The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning.

You can find all the original Gemma 3n checkpoints under the [Gemma 3n][gemma3n-collection] release.

> [!TIP]
> Click on the Gemma 3n models in the right sidebar for more examples of how to apply Gemma to different vision, audio,
> and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="google/gemma-3n-e4b",
    device=0,
    dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

model = Gemma3nForConditionalGeneration.from_pretrained(
    "google/gemma-3n-e4b-it",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "google/gemma-3n-e4b-it",
    padding_side="left"
)

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
).to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model google/gemma-3n-e2b --device 0
```

</hfoption>
</hfoptions>

## Notes

-   Use [`Gemma3nForConditionalGeneration`] for image-audio-and-text, image-and-text, image-and-audio, audio-and-text,
    image-only and audio-only inputs.
-   Gemma 3n supports multiple images per input, but make sure the images are correctly batched before passing them to
    the processor. Each batch should be a list of one or more images.

    ```py
    url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    messages =[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url_cow},
                {"type": "image", "url": url_cat},
                {"type": "text", "text": "Which image is cuter?"},
            ]
        },
    ]
    ```
-   Text passed to the processor should have a `<image_soft_token>` token wherever an image should be inserted.
-   Gemma 3n accept at most one target audio clip per input, though multiple audio clips can be provided in few-shot
    prompts, for example.
-   Text passed to the processor should have a `<audio_soft_token>` token wherever an audio clip should be inserted.
-   The processor has its own [`~ProcessorMixin.apply_chat_template`] method to convert chat messages to model inputs.

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
[laurel]: https://arxiv.org/abs/2411.07501
[matformer]: https://arxiv.org/abs/2310.07707
[spark-transformer]: https://arxiv.org/abs/2506.06644
[usm]: https://arxiv.org/abs/2303.01037
