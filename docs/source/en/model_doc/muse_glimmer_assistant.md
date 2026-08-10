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

# MuseGlimmerAssistant

[MuseGlimmerAssistant](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model) is the [DFlash](https://huggingface.co/papers/2602.06036) drafter for [MuseGlimmer](./muse_glimmer). It is not a standalone language model. It has 5 sliding window layers and no embeddings of its own. It borrows the main model's input and output embeddings, and reads the main model's hidden states at `target_layer_ids` (layers 1, 13, 25, 37, and 49 by default) as context.

Rather than drafting one token at a time, the drafter denoises a whole block of `block_size` masked tokens in a single forward pass, like a diffusion window. The main model then verifies the block in one step. Meta reports 3.1x faster decoding on an RTX 5090 and 1.5-1.8x on Apple M-series chips.

Pass the drafter to [`~GenerationMixin.generate`] as `assistant_model` and set `speculation_type="dflash"`. The drafter must be loaded in the same dtype and on the same device as the main model.

```python
from transformers import AutoProcessor, MuseGlimmerAssistantModel, MuseGlimmerForConditionalGeneration

processor = AutoProcessor.from_pretrained("meta-models/Muse-Glimmer-30B")
model = MuseGlimmerForConditionalGeneration.from_pretrained(
    "meta-models/Muse-Glimmer-30B",
    device_map="auto",
)
drafter = MuseGlimmerAssistantModel.from_pretrained(
    "meta-models/Muse-Glimmer-30B-assistant",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Write a bash one-liner that counts lines of Python in a repo."}],
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

outputs = model.generate(
    **inputs,
    assistant_model=drafter,
    speculation_type="dflash",
    max_new_tokens=256,
)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
print(response)
```

## Notes

- The drafter needs the main model's hidden states, so `generate` forces `output_hidden_states=True` for the target model when `speculation_type="dflash"`.

## MuseGlimmerAssistantConfig

[[autodoc]] MuseGlimmerAssistantConfig


## MuseGlimmerAssistantPreTrainedModel

[[autodoc]] MuseGlimmerAssistantPreTrainedModel

## MuseGlimmerAssistantModel

[[autodoc]] MuseGlimmerAssistantModel
    - forward
