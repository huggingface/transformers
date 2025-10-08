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
*This model was released on 2024-08-23 and added to Hugging Face Transformers on 2025-02-14 and contributed by [mayank-mishra](https://huggingface.co/mayank-mishra), [shawntan](https://huggingface.co/shawntan), and [SukritiSharma](https://huggingface.co/SukritiSharma).*

# GraniteMoeShared

[GraniteMoeShared](https://huggingface.co/papers/2408.13359) adds shared experts for the mixture-of-experts (MoE).

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ibm-research/moe-7b-1b-active-shared-experts", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ibm-research/moe-7b-1b-active-shared-experts")
model = AutoModelForCausalLM.from_pretrained("ibm-research/moe-7b-1b-active-shared-experts", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## GraniteMoeSharedConfig

[[autodoc]] GraniteMoeSharedConfig

## GraniteMoeSharedModel

[[autodoc]] GraniteMoeSharedModel
    - forward

## GraniteMoeSharedForCausalLM

[[autodoc]] GraniteMoeSharedForCausalLM
    - forward