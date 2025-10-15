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

*This model was released on 2024-12-13 and added to Hugging Face Transformers on 2025-10-07 and contributed by [itazap](https://huggingface.co/itazap).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Byte Latent Transformer (BLT)

[Byte Latent Transformer](https://huggingface.co/papers/2412.09871) is a byte-level LLM architecture that matches tokenization-based LLM performance at scale. It encodes bytes into dynamically sized patches based on entropy, optimizing compute and model capacity where data complexity is higher. This approach improves inference efficiency and robustness, with the first flop-controlled scaling study up to 8B parameters and 4T training bytes. BLT demonstrates better scaling than tokenization-based models by dynamically selecting long patches for predictable data, enhancing reasoning and long-tail generalization.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="itazap/blt-1b-hf", dtype="auto")
pipeline("Plants generate energy through a process known as  ")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("itazap/blt-1b-hf", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("itazap/blt-1b-hf")

inputs = tokenizer("Plants generate energy through a process known as  ", return_tensors='pt', return_token_type_ids=False)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

</hfoption>
</hfoptions>

## BltConfig

[[autodoc]] BltConfig

[[autodoc]] BltModel
    - forward

## BltForCausalLM

[[autodoc]] BltForCausalLM
    - forward

