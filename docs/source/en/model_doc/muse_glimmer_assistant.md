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

[Muse Glimmer Assistant](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) is the speculative-decoding drafter shipped with [MuseGlimmer](./muse_glimmer). It implements [DFlash](https://huggingface.co/papers/2602.06036): instead of proposing one token at a time, it denoises a block of 16 tokens in a single forward pass, which the target model then verifies in parallel, leaving output quality unchanged. It is small — 5 decoder layers, sliding-window attention (2048) on every layer, 32 query heads and 8 key/value heads.

[`MuseGlimmerAssistantModel`] does not read token ids. It is conditioned on `noise_embeds` (the last generated token plus the mask tokens to denoise) and `context_hidden_states` (hidden states from layers `[1, 13, 25, 37, 49]` of the target, concatenated). Both come from a running target model, so the drafter is a building block rather than a standalone model, and it is not wired into [`~GenerationMixin.generate`] — the speculative-decoding loop lives in the serving stack.

```python
from transformers import AutoModel

drafter = AutoModel.from_pretrained("meta-models/Muse-Glimmer-30B-assistant", device_map="auto")
```

## MuseGlimmerAssistantConfig

[[autodoc]] MuseGlimmerAssistantConfig


## MuseGlimmerAssistantPreTrainedModel

[[autodoc]] MuseGlimmerAssistantPreTrainedModel

## MuseGlimmerAssistantModel

[[autodoc]] MuseGlimmerAssistantModel
    - forward
