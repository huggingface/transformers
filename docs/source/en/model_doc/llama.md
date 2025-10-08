<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-02-27 and added to Hugging Face Transformers on 2023-03-16 and contributed by [zphang](https://huggingface.co/zphang) and [BlackSamorez](https://huggingface.co/BlackSamorez).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Llama

[LLaMA: Open and Efficient Foundation Language Models](https://huggingface.co/papers/2302.13971) presents a series of foundation language models sized from 7B to 65B parameters. Trained on trillions of tokens, LLaMA demonstrates the capability to achieve state-of-the-art results using only publicly available datasets. Specifically, LLaMA-13B surpasses GPT-3 (175B) on most benchmarks, while LLaMA-65B competes with top models like Chinchilla-70B and PaLM-540B. All models are released to the research community.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="huggyllama/llama-7b", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## LlamaConfig

[[autodoc]] LlamaConfig

## LlamaTokenizer

[[autodoc]] LlamaTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## LlamaTokenizerFast

[[autodoc]] LlamaTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## LlamaModel

[[autodoc]] LlamaModel
    - forward

## LlamaForCausalLM

[[autodoc]] LlamaForCausalLM
    - forward

## LlamaForSequenceClassification

[[autodoc]] LlamaForSequenceClassification
    - forward

## LlamaForQuestionAnswering

[[autodoc]] LlamaForQuestionAnswering
    - forward

## LlamaForTokenClassification

[[autodoc]] LlamaForTokenClassification
    - forward

