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
*This model was released on 2019-10-29 and added to Hugging Face Transformers on 2020-11-16 and contributed by [sshleifer](https://huggingface.co/sshleifer).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BART

[BART](https://huggingface.co/papers/1910.13461) is a Transformer-based sequence-to-sequence model trained as a denoising autoencoder: text is corrupted with noise and the model learns to reconstruct the original. Its architecture combines a bidirectional encoder like BERT with a left-to-right decoder like GPT, making it a general framework for many pretraining approaches. Using techniques like sentence shuffling and span in-filling, BART achieves strong results on both generation and comprehension tasks, matching RoBERTa on GLUE and SQuAD while setting new state-of-the-art results in summarization, dialogue, and question answering. It also boosts machine translation performance and allows ablation experiments that replicate and compare other pretraining schemes.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="summarization", model="facebook/bart-large-cnn", dtype="auto")
pipeline("The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

text="""
The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.
"""
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

</hfoption>
</hfoptions>

## BartConfig

[[autodoc]] BartConfig
    - all

## BartTokenizer

[[autodoc]] BartTokenizer
    - all

## BartTokenizerFast

[[autodoc]] BartTokenizerFast
    - all

## BartModel

[[autodoc]] BartModel
    - forward

## BartForConditionalGeneration

[[autodoc]] BartForConditionalGeneration
    - forward

## BartForSequenceClassification

[[autodoc]] BartForSequenceClassification
    - forward

## BartForQuestionAnswering

[[autodoc]] BartForQuestionAnswering
    - forward

## BartForCausalLM

[[autodoc]] BartForCausalLM
    - forward

