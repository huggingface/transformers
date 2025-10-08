
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
*This model was released on 2025-04-08 and added to Hugging Face Transformers on 2025-06-25.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# T5Gemma

[T5Gemma](https://huggingface.co/papers/2504.06225) investigates adapting pretrained decoder-only large language models (LLMs) into encoder-decoder architectures to combine the strengths of both approaches. The authors explore various pretraining objectives, parameter initialization, and optimization strategies to enable adaptation without training from scratch. Experiments on Gemma 2 (2B and 9B) and newly pretrained mT5-sized models (up to 1.6B) show that adapted encoder-decoder LLMs achieve comparable pretraining performance, substantially better finetuning results, and improved benchmarks like SuperGLUE, all under similar inference budgets. The approach also supports flexible model combinations, with larger encoders boosting performance, and checkpoints will be released for future research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="text2text-generation", model="google/t5gemma-2b-2b-prefixlm-it", dtype="auto",)
messages = [
    {"role": "user", "content": "How do plants create energy?"},
]
pipeline(messages)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-prefixlm-it")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2b-2b-prefixlm-it", dtype="auto",)

messages = [
    {"role": "user", "content": "How do plants create energy?"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

</hfoption>
</hfoptions>

## T5GemmaConfig

[[autodoc]] T5GemmaConfig

## T5GemmaModuleConfig

[[autodoc]] T5GemmaModuleConfig

## T5GemmaModel

[[autodoc]] T5GemmaModel
    - forward

## T5GemmaEncoderModel

[[autodoc]] T5GemmaEncoderModel
    - forward

## T5GemmaForConditionalGeneration

[[autodoc]] T5GemmaForConditionalGeneration
    - forward

## T5GemmaForSequenceClassification

[[autodoc]] T5GemmaForSequenceClassification
    - forward

## T5GemmaForTokenClassification

[[autodoc]] T5GemmaForTokenClassification
    - forward
