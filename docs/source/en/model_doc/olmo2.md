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
*This model was released on 2024-02-01 and added to Hugging Face Transformers on 2024-11-25 and contributed by [shanearora](https://huggingface.co/shanearora).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMo2

[OLMo2](https://huggingface.co/papers/2501.00656) is the next-generation fully open language model series, featuring dense autoregressive architectures with improved training stability and per-token efficiency. It introduces a new pretraining data mixture, Dolmino Mix 1124, which enhances downstream task performance when applied in late-stage curriculum training. The OLMo 2-Instruct variant incorporates permissive instruction data and reinforcement learning with verifiable rewards (RLVR), following best practices from T"ulu 3. Models at 7B and 13B scales are fully open, competitive with or surpassing comparable open-weight models like Llama 3.1 and Qwen 2.5, and all code, data, and checkpoints are publicly released.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="allenai/OLMo-2-0425-1B", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- OLMo2 uses RMSNorm instead of standard layer norm. RMSNorm is applied to attention queries and keys. It's applied after the attention and feedforward layers rather than before.
- OLMo2 requires Transformers v4.48 or higher.
- Load specific intermediate checkpoints by adding the `revision` parameter to [`~AutoModel.from_pretrained`].

## Olmo2Config

[[autodoc]] Olmo2Config

## Olmo2Model

[[autodoc]] Olmo2Model
    - forward

## Olmo2ForCausalLM

[[autodoc]] Olmo2ForCausalLM
    - forward
