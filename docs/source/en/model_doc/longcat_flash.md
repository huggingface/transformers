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

*This model was released on 2025-09-01 and added to Hugging Face Transformers on 2025-10-07 and contributed by [Molbap](https://huggingface.co/Molbap).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# LongCatFlash

[LongCat-Flash](https://huggingface.co/papers/2509.01322) is a 560B parameter Mixture-of-Experts (MoE) model that dynamically activates 18.6B-31.3B parameters (average ~27B) based on context. It features a shortcut-connected architecture that enhances inference speed to over 100 tokens/second and achieves high accuracy (89.71% on MMLU) and strong agentic tool use capabilities. The model supports up to 128k context length and is optimized for reasoning, coding, and tool-calling tasks.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meituan-longcat/LongCat-Flash-Chat")
model = AutoModelForCausalLM.from_pretrained("meituan-longcat/LongCat-Flash-Chat", dtype="auto", tp_plan="auto")

chat = [
      {"role": "user", "content": "How do plants generate energy?"},
]

inputs = tokenizer.apply_chat_template(
      chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
outputs = model.generate(inputs, max_new_tokens=30)
print(tokenizer.batch_decode(outputs))
```

</hfoption>
</hfoptions>

## LongcatFlashConfig

[[autodoc]] LongcatFlashConfig

## LongcatFlashPreTrainedModel

[[autodoc]] LongcatFlashPreTrainedModel
    - forward

## LongcatFlashModel

[[autodoc]] LongcatFlashModel
    - forward

## LongcatFlashForCausalLM

[[autodoc]] LongcatFlashForCausalLM

