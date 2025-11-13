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
*This model was released on 2024-09-03 and added to Hugging Face Transformers on 2024-09-03 and contributed by [Muennighoff](https://hf.co/Muennighoff).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# OLMoE

[OLMoE: Open Mixture-of-Experts Language Models](https://huggingface.co/papers/2409.02060) is an open-source, state-of-the-art language model using a sparse Mixture-of-Experts (MoE) architecture. Its largest variant, OLMoE-1B-7B, has 7 billion parameters but activates only 1 billion per input token, and it was pretrained on 5 trillion tokens. The model was further adapted into an instruction-following version, OLMoE-1B-7B-Instruct, and outperforms comparable models, even exceeding larger ones like Llama2-13B-Chat. The authors analyze expert routing and specialization in MoE training and release all model weights, training data, code, and logs publicly.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="allenai/OLMoE-1B-7B-0125", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0125")
model = AutoModelForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0125", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>e("The future of artificial intelligence is")
```

## OlmoeConfig

[[autodoc]] OlmoeConfig

## OlmoeModel

[[autodoc]] OlmoeModel
    - forward

## OlmoeForCausalLM

[[autodoc]] OlmoeForCausalLM
    - forward

