<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-07-18 and added to Hugging Face Transformers on 2023-07-18 and contributed by [ArthurZ](https://huggingface.co/ArthurZ) and [lysandre](https://huggingface.co/lysandre).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Llama 2

[Llama 2](https://huggingface.co/papers/2307.09288) is a series of pretrained and fine-tuned large language models with parameter sizes ranging from 7 billion to 70 billion. The fine-tuned versions, Llama 2-Chat, are specifically optimized for chat applications. These models demonstrate superior performance compared to other open-source chat models across various benchmarks and human evaluations for helpfulness and safety, potentially serving as viable alternatives to proprietary models. The paper details the fine-tuning process and safety enhancements applied to Llama 2-Chat, encouraging further community contributions to the responsible advancement of large language models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="meta-llama/Llama-2-7b-hf", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto",)

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

