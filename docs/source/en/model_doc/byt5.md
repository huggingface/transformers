<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
<div style="float: right;">
  <div class="flex flex-wrap space-x-1">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
    <img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=flax&logoColor=white">
  </div>
</div>

# ByT5

[ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) was proposed by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel.

ByT5 is a version of the T5 model that works directly on **raw UTF-8 bytes** instead of using tokens or subwords. That means no tokenizer is needed — it just takes raw text, byte by byte. This makes it naturally multilingual, more robust to spelling/noise, and simpler to use, since there's no complex text preprocessing.

You can find all the original ByT5 checkpoints under the [ByT5 model collection](https://huggingface.co/google).

> [!TIP]
> Click on the ByT5 models in the right sidebar for more examples of how to apply ByT5 to different text generation and translation tasks.

The example below demonstrates how to run ByT5 using either [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/byt5-small")
result = pipe("Translate English to French: Life is beautiful.")
print(result)
```

</hfoption>
<hfoption id="AutoModel">

```python
from transformers import T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

num_special_tokens = 3  # ByT5 reserves 0, 1, 2 for special tokens
input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens
labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

loss = model(input_ids, labels=labels).loss
print(loss.item())
```

```python
# With tokenizer for batching:
from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

inputs = tokenizer(["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt")
labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids

loss = model(**inputs, labels=labels).loss
print(loss.item())
```

</hfoption>
</hfoptions>

## Quantization

Quantization reduces the memory burden of large models by using lower-precision weights. While ByT5 comes in relatively small versions (`small`, `base`, etc.), quantization can still be applied using libraries like `bitsandbytes` or `optimum`. For more on this, refer to the [Quantization overview](../quantization/overview).

Example:
```python
# Use with 8-bit loading
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small", load_in_8bit=True, device_map="auto")
```

## Attention Visualization

Use the [`AttentionMaskVisualizer`](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/attention_visualizer.py) to better understand what tokens the model can and cannot attend to:

```python
from transformers.utils.attention_visualizer import AttentionMaskVisualizer

visualizer = AttentionMaskVisualizer("google/byt5-small")
visualizer("Translate English to French: Life is beautiful.")
```

## Notes

- ByT5 uses **no traditional tokenizer**. Instead, it operates directly on UTF-8 bytes. That means the input must be properly encoded, and users must account for the three special tokens (IDs 0, 1, 2).
- Masking is also unique: instead of sentinel tokens like `{extra_id_0}`, ByT5 uses the top byte values (e.g. 258, 257...) for masking.

```python
# Example: character-level denoising with mask tokens
input_ids = tokenizer("The dog chases a ball in the park.").input_ids
masked_input = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
output = model.generate(masked_input, max_length=100)
```

ByT5 was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original implementation is available on [GitHub](https://github.com/google-research/byt5).

## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer