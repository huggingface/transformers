<!--Copyright 2025 NXAI GmbH. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-25 and contributed by [NX-AI](https://huggingface.co/NX-AI).*

# xLSTM

[xLSTM: Extended Long Short-Term Memory](https://huggingface.co/papers/2405.04517) updates the original LSTM architecture by introducing exponential gating, matrix memory expansion, and parallelizable training to compete with Transformer models. The model modifies the LSTM memory structure to include scalar and matrix memory variants, enhancing its performance and scalability. Exponential gating and these memory modifications enable xLSTM to match or outperform state-of-the-art Transformers and State Space Models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="NX-AI/xLSTM-7b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7b")
model = AutoModelForCausalLM.from_pretrained("NX-AI/xLSTM-7b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## xLSTMConfig

[[autodoc]] xLSTMConfig

## xLSTMModel

[[autodoc]] xLSTMModel
    - forward

## xLSTMLMHeadModel

[[autodoc]] xLSTMForCausalLM
    - forward

