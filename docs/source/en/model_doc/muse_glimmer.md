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
*This model was contributed to Hugging Face Transformers on 2026-08-09.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MuseGlimmer

[MuseGlimmer](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) is a 30B multimodal model from Meta Superintelligence Lab, built for agents that run locally on consumer hardware. A dense 52-layer text decoder handles interleaved text and images, and a frozen ViT-G/14 perception encoder turns screenshots, charts, and documents into visual tokens. Output is text only.

Three out of every four decoder layers use sliding window attention over a 2048-token window. The fourth is a full attention layer with rotary embeddings disabled (NoPE), giving the model a 131K context. Attention also softcaps the final logits and applies an extra scale to the queries after QK-norm.

The model ships with a companion drafter, [MuseGlimmerAssistant](./muse_glimmer_assistant), for DFlash speculative decoding, which drafts a whole block of tokens per forward pass.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="meta-models/Muse-Glimmer-30B",
    dtype="auto",
    device_map="auto",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
pipeline(messages, max_new_tokens=64)
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B")
model = AutoModelForMultimodalLM.from_pretrained(
    "meta-models/Muse-Glimmer-30B",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
```

</hfoption>
</hfoptions>

## Notes

- The chat template accepts a `reasoning_strength` kwarg to trade quality against latency. Pass it through `apply_chat_template` along with any tool definitions.

    ```python
    inputs = processor.apply_chat_template(
        messages,
        reasoning_strength="high",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    ```

- Videos are processed as frames, and [`MuseGlimmerProcessor`] writes a `Time: <seconds>s` marker before each temporal group so the model can reason about ordering. The timestamps come from the video metadata, so pass `video_metadata` when the frame rate can't be inferred. Otherwise the processor warns and falls back to 24 fps, which shifts every timestamp in the prompt.
- Images and videos are expanded into token spans by the processor. An image becomes `<|image_start|>` followed by one `<|patch|>` per merged patch and `<|image_end|>`. Only include `{"type": "image"}` in the chat messages.
- [`MuseGlimmerTextConfig`] derives `layer_types` and `layer_rope_theta` from `num_hidden_layers` in its `__post_init__`, counting the NoPE layers backward from the last layer. Set both explicitly if you change the layer count and want a different pattern.
- See the [Meta is back with Muse Glimmer: local, agentic, multimodal, and open source!](https://huggingface.co/blog/muse-glimmer) blog post for more details and example usage.

## MuseGlimmerConfig

[[autodoc]] MuseGlimmerConfig

## MuseGlimmerTextConfig

[[autodoc]] MuseGlimmerTextConfig

## MuseGlimmerVisionConfig

[[autodoc]] MuseGlimmerVisionConfig

## MuseGlimmerImageProcessor

[[autodoc]] MuseGlimmerImageProcessor

## MuseGlimmerVideoProcessor

[[autodoc]] MuseGlimmerVideoProcessor

## MuseGlimmerProcessor

[[autodoc]] MuseGlimmerProcessor

## MuseGlimmerPreTrainedModel

[[autodoc]] MuseGlimmerPreTrainedModel

## MuseGlimmerTextModel

[[autodoc]] MuseGlimmerTextModel
    - forward

## MuseGlimmerVisionModel

[[autodoc]] MuseGlimmerVisionModel
    - forward

## MuseGlimmerModel

[[autodoc]] MuseGlimmerModel
    - forward

## MuseGlimmerForConditionalGeneration

[[autodoc]] MuseGlimmerForConditionalGeneration
    - forward