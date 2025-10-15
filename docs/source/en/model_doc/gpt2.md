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
*This model was released on 2019-02-14 and added to Hugging Face Transformers on 2020-11-16 and contributed by [thomwolf](https://huggingface.co/thomwolf).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GPT2

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) is a 1.5 billion parameter Transformer that demonstrates strong zero-shot learning, reaching 55 F1 on the CoQA dataset without using its training examples and achieving state-of-the-art results on 7 of 8 language modeling benchmarks. Performance scales log-linearly with model capacity, though even the largest model still underfits the dataset. The results highlight that scaling up language models enables coherent text generation and effective transfer across diverse tasks from naturally occurring data.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text-generation", model="openai-community/gpt2", dtype="auto",)
pipeline("Plants create energy through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", dtype="auto",)

inputs = tokenizer("Plants create energy through a process known as photosynthesis.", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## Usage tips

- Pad inputs on the right. GPT-2 uses absolute position embeddings.
- GPT-2 reuses previously computed key-value attention pairs. Access this feature with the `past_key_values` parameter in [`GPT2Model.forward`].
- Enable the [`GPT2Config.scale_attn_by_inverse_layer_idx`] and [`GPT2Config.reorder_and_upcast_attn`] parameters to apply training stability improvements from Mistral.

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2 specific outputs

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

## GPT2Model
[[autodoc]] GPT2Model
    - forward

## GPT2LMHeadModel

[[autodoc]] GPT2LMHeadModel
    - forward

## GPT2DoubleHeadsModel

[[autodoc]] GPT2DoubleHeadsModel
    - forward

## GPT2ForQuestionAnswering

[[autodoc]] GPT2ForQuestionAnswering
    - forward

## GPT2ForSequenceClassification

[[autodoc]] GPT2ForSequenceClassification
    - forward

## GPT2ForTokenClassification

[[autodoc]] GPT2ForTokenClassification
    - forward

