<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-03-28 and added to Hugging Face Transformers on 2024-04-18.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Jamba

[Jamba](https://huggingface.co/papers/2403.19887) is a large language model that combines Transformer layers with Mamba layers in a hybrid mixture-of-experts (MoE) design, balancing efficiency and capacity. By selectively adding MoE to certain layers, the model scales up while keeping active parameters manageable, enabling deployment on a single 80GB GPU. This architecture achieves high throughput, reduced memory usage, and strong benchmark performance, including handling context lengths up to 256K tokens. The work also explores key architectural tradeoffs in layer mixing and expert integration, and the team is releasing weights and ablation checkpoints for further research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="ai21labs/AI21-Jamba-Mini-1.6", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6")
model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Mini-1.6", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Don't quantize the Mamba blocks. This prevents model performance degradation.
- Use optimized Mamba kernels for better performance. Mamba without kernels results in significantly lower latencies. Set `use_mamba_kernels=False` in [`~AutoModel.from_pretrained`] if you need to disable kernels.

## JambaConfig

[[autodoc]] JambaConfig

## JambaModel

[[autodoc]] JambaModel
    - forward

## JambaForCausalLM

[[autodoc]] JambaForCausalLM
    - forward

## JambaForSequenceClassification

[[autodoc]] transformers.JambaForSequenceClassification
    - forward

