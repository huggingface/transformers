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

# ByT5

## Overview

The ByT5 model was presented in [ByT5: Towards a token-free future with pre-trained byte-to-byte models](https://arxiv.org/abs/2105.13626) by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir
Kale, Adam Roberts, Colin Raffel.

The abstract from the paper is the following:

*Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units.
Encoding text as a sequence of tokens requires a tokenizer, which is typically created as an independent artifact from
the model. Token-free models that instead operate directly on raw text (bytes or characters) have many benefits: they
can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by
removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token
sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of
operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with
minimal modifications to process byte sequences. We carefully characterize the trade-offs in terms of parameter count,
training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level
counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on
tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of
pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our
experiments.*

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The original code can be
found [here](https://github.com/google-research/byt5).

ByT5's architecture is based on the T5v1.1 model, so one can refer to [T5v1.1's documentation page](t5v1.1). They
only differ in how inputs should be prepared for the model, see the code examples below.

Since ByT5 was pre-trained unsupervisedly, there's no real advantage to using a task prefix during single-task
fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.


### Example

ByT5 works on raw UTF-8 bytes, so it can be used without a tokenizer:

```python
>>> from transformers import T5ForConditionalGeneration
>>> import torch

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

>>> num_special_tokens = 3
>>> # Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
>>> # => Need to shift utf-8 character encodings by 3 before passing ids to model.

>>> input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

>>> labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

>>> loss = model(input_ids, labels=labels).loss
>>> loss.item()
2.66
```

For batched inference and training it is however recommended to make use of the tokenizer:

```python
>>> from transformers import T5ForConditionalGeneration, AutoTokenizer

>>> model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

>>> model_inputs = tokenizer(
...     ["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt"
... )
>>> labels_dict = tokenizer(
...     ["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt"
... )
>>> labels = labels_dict.input_ids

>>> loss = model(**model_inputs, labels=labels).loss
>>> loss.item()
17.9
```

Similar to [T5](t5), ByT5 was trained on the span-mask denoising task. However, 
since the model works directly on characters, the pretraining task is a bit 
different. Let's corrupt some characters of the 
input sentence `"The dog chases a ball in the park."` and ask ByT5 to predict them 
for us.

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-base")

>>> input_ids_prompt = "The dog chases a ball in the park."
>>> input_ids = tokenizer(input_ids_prompt).input_ids

>>> # Note that we cannot add "{extra_id_...}" to the string directly
>>> # as the Byte tokenizer would incorrectly merge the tokens
>>> # For ByT5, we need to work directly on the character level
>>> # Contrary to T5, ByT5 does not use sentinel tokens for masking, but instead
>>> # uses final utf character ids.
>>> # UTF-8 is represented by 8 bits and ByT5 has 3 special tokens.
>>> # => There are 2**8+2 = 259 input ids and mask tokens count down from index 258.
>>> # => mask to "The dog [258]a ball [257]park."

>>> input_ids = torch.tensor([input_ids[:8] + [258] + input_ids[14:21] + [257] + input_ids[28:]])
>>> input_ids
tensor([[ 87, 107, 104,  35, 103, 114, 106,  35, 258,  35, 100,  35, 101, 100, 111, 111, 257,  35, 115, 100, 117, 110,  49,   1]])

>>> # ByT5 produces only one char at a time so we need to produce many more output characters here -> set `max_length=100`.
>>> output_ids = model.generate(input_ids, max_length=100)[0].tolist()
>>> output_ids
[0, 258, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118, 257,  35, 108, 113,  35, 119, 107, 104,  35, 103, 108, 118, 102, 114, 256, 108, 113,  35, 119, 107, 104, 35, 115, 100, 117, 110,  49,  35,  87, 107, 104,  35, 103, 114, 106, 35, 108, 118,  35, 119, 107, 104,  35, 114, 113, 104,  35, 122, 107, 114,  35, 103, 114, 104, 118,  35, 100,  35, 101, 100, 111, 111,  35, 108, 113, 255,  35, 108, 113,  35, 119, 107, 104,  35, 115, 100, 117, 110,  49]

>>> # ^- Note how 258 descends to 257, 256, 255

>>> # Now we need to split on the sentinel tokens, let's write a short loop for this
>>> output_ids_list = []
>>> start_token = 0
>>> sentinel_token = 258
>>> while sentinel_token in output_ids:
...     split_idx = output_ids.index(sentinel_token)
...     output_ids_list.append(output_ids[start_token:split_idx])
...     start_token = split_idx
...     sentinel_token -= 1

>>> output_ids_list.append(output_ids[start_token:])
>>> output_string = tokenizer.batch_decode(output_ids_list)
>>> output_string
['<pad>', 'is the one who does', ' in the disco', 'in the park. The dog is the one who does a ball in', ' in the park.']
```


## ByT5Tokenizer

[[autodoc]] ByT5Tokenizer

See [`ByT5Tokenizer`] for all details.
