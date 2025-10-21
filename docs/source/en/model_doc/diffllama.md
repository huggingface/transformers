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
*This model was released on 2024-10-07 and added to Hugging Face Transformers on 2025-01-07.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DiffLlama

[DiffLlama](https://huggingface.co/papers/2410.05258) integrates the Llama model with Differential Transformer's Attention mechanism. This differential attention calculates scores as the difference between two softmax attention maps, reducing noise and promoting sparse attention. Experiments demonstrate that DiffLlama outperforms traditional Transformer models in scaling, long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and activation outlier reduction. It enhances accuracy and robustness in in-context learning and reduces distractions from irrelevant context, improving performance in question answering and text summarization.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="kajuma/DiffLlama-0.3B-handcut", dtype="auto")
pipeline("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("kajuma/DiffLlama-0.3B-handcut", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("kajuma/DiffLlama-0.3B-handcut")

inputs = tokenizer("植物は光合成と呼ばれる過程を通じてエネルギーを作り出します。", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## DiffLlamaConfig

[[autodoc]] DiffLlamaConfig

## DiffLlamaModel

[[autodoc]] DiffLlamaModel
    - forward

## DiffLlamaForCausalLM

[[autodoc]] DiffLlamaForCausalLM
    - forward

## DiffLlamaForSequenceClassification

[[autodoc]] DiffLlamaForSequenceClassification
    - forward

## DiffLlamaForQuestionAnswering

[[autodoc]] DiffLlamaForQuestionAnswering
    - forward

## DiffLlamaForTokenClassification

[[autodoc]] DiffLlamaForTokenClassification
    - forward

