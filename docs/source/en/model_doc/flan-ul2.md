<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2023-03-03 and added to Hugging Face Transformers on 2023-06-20.*

# FLAN-UL2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[Flan-UL2](https://www.yitay.net/blog/flan-ul2-20b) is an encoder decoder model based on the T5 architecture. It uses the same configuration as the [UL2](ul2) model released earlier last year.
It was fine tuned using the "Flan" prompt tuning and dataset collection. Similar to `Flan-T5`,  one can directly use FLAN-UL2 weights without finetuning the model:

According to the original blog here are the notable improvements:

- The original UL2 model was only trained with receptive field of 512, which made it non-ideal for N-shot prompting where N is large.
- The Flan-UL2 checkpoint uses a receptive field of 2048 which makes it more usable for few-shot in-context learning.
- The original UL2 model also had mode switch tokens that was rather mandatory to get good performance. However, they were a little cumbersome as this requires often some changes during inference or finetuning. In this update/change, we continue training UL2 20B for an additional 100k steps (with small batch) to forget “mode tokens” before applying Flan instruction tuning. This Flan-UL2 checkpoint does not require mode tokens anymore.
Google has released the following variants:

The original checkpoints can be found [here](https://github.com/google-research/google-research/tree/master/ul2).


## Running on low resource devices

The model is pretty heavy (~40GB in half precision) so if you just want to run the model, make sure you load your model in 8bit, and use `device_map="auto"` to make sure  you don't have any OOM issue!

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", load_in_8bit=True, device_map="auto")
>>> tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

>>> inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['In a large skillet, brown the ground beef and onion over medium heat. Add the garlic']
```

<Tip>

Refer to [T5's documentation page](t5) for API reference, tips, code examples and notebooks.

</Tip>
