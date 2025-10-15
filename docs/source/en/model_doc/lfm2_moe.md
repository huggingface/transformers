<!--Copyright 2025 the HuggingFace Team. All rights reserved.

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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-10-07.*

# Lfm2Moe

LFM2-MoE is a Mixture-of-Experts version of the LFM2 architecture, designed for efficient on-device inference. It combines gated convolutions for local context with grouped-query attention (GQA) for efficient global reasoning. By adding sparse MoE feed-forward layers, it boosts representational power while keeping computational costs low. The initial model, LFM2-8B-A1B, has 8.3B total parameters with 1.5B active per inference, matching the quality of 3–4B dense models while running faster than typical 1.5B models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="LiquidAI/LFM2-8B-A1B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-8B-A1B")
model = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2-8B-A1B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Lfm2MoeConfig

[[autodoc]] Lfm2MoeConfig

## Lfm2MoeForCausalLM

[[autodoc]] Lfm2MoeForCausalLM

## Lfm2MoeModel

[[autodoc]] Lfm2MoeModel
    - forward

## Lfm2MoePreTrainedModel

[[autodoc]] Lfm2MoePreTrainedModel
    - forward
