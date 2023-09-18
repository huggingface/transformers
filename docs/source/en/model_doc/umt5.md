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

# UMT5

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=umt5">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-mt5-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/mt5-small-finetuned-arxiv-cs-finetuned-arxiv-cs-full">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The UMT5 model was proposed in [UniMax: Fairer and More Effective Language Sampling for Large-Scale Multilingual Pretraining](https://openreview.net/forum?id=kXwdL1cWOAi) by Hyung Won Chung, Xavier Garcia, Adam Roberts, Yi Tay, Orhan Firat, Sharan Narang, Noah Constant.

The abstract from the paper is the following:

*Pretrained multilingual large language models have typically used heuristic temperature-based sampling to balance between different languages. However previous work has not systematically evaluated the efficacy of different pretraining language distributions across model scales. In this paper, we propose a new sampling method, UniMax, that delivers more uniform coverage of head languages while mitigating overfitting on tail languages by explicitly capping the number of repeats over each language's corpus. We perform an extensive series of ablations testing a range of sampling strategies on a suite of multilingual benchmarks, while varying model scale. We find that UniMax outperforms standard temperature-based sampling, and the benefits persist as scale increases. As part of our contribution, we release: (i) an improved and refreshed mC4 multilingual corpus consisting of 29 trillion characters across 107 languages, and (ii) a suite of pretrained umT5 model checkpoints trained with UniMax sampling.*

Tips: 

- UMT5 was only pre-trained on [mC4](https://huggingface.co/datasets/mc4) excluding any supervised training.
Therefore, this model has to be fine-tuned before it is usable on a downstream task, unlike the original T5 model.
- Since umT5 was pre-trained in an unsupervise manner, there's no real advantage to using a task prefix during single-task
fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

Google has released the following variants:

- [google/umt5-small](https://huggingface.co/google/umt5-small)
- [google/umt5-base](https://huggingface.co/google/umt5-base)
- [google/umt5-xl](https://huggingface.co/google/umt5-xl)
- [google/umt5-xxl](https://huggingface.co/google/umt5-xxl).

This model was contributed by [agemagician](https://huggingface.co/agemagician) and [stefan-it](https://huggingface.co/stefan-it). The original code can be
found [here](https://github.com/google-research/t5x).

One can refer to [T5's documentation page](t5) for more tips, code examples and notebooks.

## Differences with mT5?
`UmT5` is based on mT5, with a non-shared relative positional bias that is computed for each layer. This means that the model set `has_relative_bias` for each layer.
The conversion script is also different because the model was saved in t5x's latest checkpointing format.

# Sample usage

```python
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> model = AutoModelForSeq2SeqLM.from_pretrained("google/umt5-small")
>>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")

>>> inputs = tokenizer(
...     "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.",
...     return_tensors="pt",
... )
>>> outputs = model.generate(**inputs)
>>> print(tokenizer.batch_decode(outputs))
['<pad><extra_id_0>nyone who<extra_id_1> drink<extra_id_2> a<extra_id_3> alcohol<extra_id_4> A<extra_id_5> A. This<extra_id_6> I<extra_id_7><extra_id_52><extra_id_53></s>']
```

## UMT5Config

[[autodoc]] UMT5Config

## UMT5Model

[[autodoc]] UMT5Model
    - forward

## UMT5ForConditionalGeneration

[[autodoc]] UMT5ForConditionalGeneration
    - forward

## UMT5EncoderModel

[[autodoc]] UMT5EncoderModel
    - forward

## UMT5ForSequenceClassification

[[autodoc]] UMT5ForSequenceClassification
    - forward

## UMT5ForQuestionAnswering

[[autodoc]] UMT5ForQuestionAnswering
    - forward

