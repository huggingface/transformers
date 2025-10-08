<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2018-09-19 and added to Hugging Face Transformers on 2023-06-20 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GPT

[GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) addresses the challenge of limited labeled data for natural language understanding tasks like textual entailment, question answering, semantic similarity, and document classification. It shows that pretraining a language model generatively on large unlabeled text, followed by task-specific discriminative fine-tuning with task-aware input transformations, can achieve strong performance with minimal architectural changes. This approach outperforms traditional discriminative models designed for individual tasks and improves the state of the art in 9 of 12 evaluated benchmarks. The results demonstrate that a single, task-agnostic model can effectively transfer knowledge across diverse NLU tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/openai-gpt", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/openai-gpt")
model = AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## OpenAIGPTConfig

[[autodoc]] OpenAIGPTConfig

## OpenAIGPTTokenizer

[[autodoc]] OpenAIGPTTokenizer
    - save_vocabulary

## OpenAIGPTTokenizerFast

[[autodoc]] OpenAIGPTTokenizerFast

## OpenAI specific outputs

[[autodoc]] models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput

## OpenAIGPTModel

[[autodoc]] OpenAIGPTModel
    - forward

## OpenAIGPTLMHeadModel

[[autodoc]] OpenAIGPTLMHeadModel
    - forward

## OpenAIGPTDoubleHeadsModel

[[autodoc]] OpenAIGPTDoubleHeadsModel
    - forward

## OpenAIGPTForSequenceClassification

[[autodoc]] OpenAIGPTForSequenceClassification
    - forward

