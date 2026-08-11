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
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# MuseGlimmer

[Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a 30B dense causal language model with a ~1.8B [perception encoder](https://huggingface.co/papers/2504.13181), distilled from Muse Spark for agentic workloads on consumer hardware. It takes interleaved text and images and returns text.

The text backbone repeats a `[local, local, local, global]` attention pattern with a 2048-token sliding window, gated attention and grouped-query attention (32 query heads, 2 key/value heads), over a 131,072-token context. The release also ships a [DFlash](https://huggingface.co/papers/2602.06036) speculative-decoding drafter, [MuseGlimmerAssistant](./muse_glimmer_assistant).

You can find the original checkpoints under the [meta-models](https://huggingface.co/meta-models) organization.

> [!TIP]
> This model was contributed by [Meta Superintelligence Lab](https://huggingface.co/meta-models).

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="meta-models/Muse-Glimmer-30B",
    device_map="auto",
)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }
]
pipeline(text=messages, max_new_tokens=128, return_full_text=False)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

processor = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B")
model = AutoModelForImageTextToText.from_pretrained("meta-models/Muse-Glimmer-30B", device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## Notes

- The chat template writes `<|begin_of_text|>` into the rendered text. Tokenizing that text in a separate step needs `add_special_tokens=False`, otherwise a second BOS is prepended. Passing `tokenize=True` to [`~ProcessorMixin.apply_chat_template`], as above, handles this for you. For raw text that did not come from the template, `add_special_tokens=True` stays correct.

- Reasoning effort is set with the `reasoning_strength` template keyword — `low`, `medium`, `high` (default) or `xhigh`.

    ```python
    processor.apply_chat_template(messages, add_generation_prompt=True, reasoning_strength="xhigh")
    ```

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
