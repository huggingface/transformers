<!--Copyright 2025 MiniMaxAI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2025-01-14 and added to Hugging Face Transformers on 2025-06-04 and contributed by [geetu040](https://github.com/geetu040) and [Shakib-IO](https://github.com/Shakib-IO).*

# MiniMax

[MiniMax-Text-01](https://huggingface.co/papers/2501.08313) is part of the MiniMax-01 series, featuring superior capabilities in processing longer contexts through lightning attention and efficient scaling. It integrates Mixture of Experts (MoE) with 32 experts and 456 billion total parameters, of which 45.9 billion are activated per token. Optimized parallel strategies and compute-communication overlap techniques enable efficient training and inference on models with hundreds of billions of parameters. MiniMax-Text-01 supports a training context length of up to 1 million tokens and can handle up to 4 million tokens during inference. The model demonstrates performance comparable to top-tier models like GPT-4o and Claude-3.5-Sonnet while offering significantly longer context windows.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="MiniMaxAI/MiniMax-Text-01-hf", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf")
model = AutoModelForCausalLM.from_pretrained("MiniMaxAI/MiniMax-Text-01-hf", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## MiniMaxConfig

[[autodoc]] MiniMaxConfig

## MiniMaxModel

[[autodoc]] MiniMaxModel
    - forward

## MiniMaxForCausalLM

[[autodoc]] MiniMaxForCausalLM
    - forward

## MiniMaxForSequenceClassification

[[autodoc]] MiniMaxForSequenceClassification
    - forward

## MiniMaxForTokenClassification

[[autodoc]] MiniMaxForTokenClassification
    - forward

## MiniMaxForQuestionAnswering

[[autodoc]] MiniMaxForQuestionAnswering
    - forward
