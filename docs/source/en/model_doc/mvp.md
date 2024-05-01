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

# MVP

## Overview

The MVP model was proposed in [MVP: Multi-task Supervised Pre-training for Natural Language Generation](https://arxiv.org/abs/2206.12131) by Tianyi Tang, Junyi Li, Wayne Xin Zhao and Ji-Rong Wen.


According to the abstract,

- MVP follows a standard Transformer encoder-decoder architecture.
- MVP is supervised pre-trained using labeled datasets.
- MVP also has task-specific soft prompts to stimulate the model's capacity in performing a certain task.
- MVP is specially designed for natural language generation and can be adapted to a wide range of generation tasks, including but not limited to summarization, data-to-text generation, open-ended dialogue system, story generation, question answering, question generation, task-oriented dialogue system, commonsense generation, paraphrase generation, text style transfer, and text simplification. Our model can also be adapted to natural language understanding tasks such as sequence classification and (extractive) question answering.

This model was contributed by [Tianyi Tang](https://huggingface.co/StevenTang). The detailed information and instructions can be found [here](https://github.com/RUCAIBox/MVP).

## Usage tips

- We have released a series of models [here](https://huggingface.co/models?filter=mvp), including MVP, MVP with task-specific prompts, and multi-task pre-trained variants.
- If you want to use a model without prompts (standard Transformer), you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')`.
- If you want to use a model with task-specific prompts, such as summarization, you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')`.
- Our model supports lightweight prompt tuning following [Prefix-tuning](https://arxiv.org/abs/2101.00190) with method `set_lightweight_tuning()`.

## Usage examples

For summarization, it is an example to use MVP and MVP with summarization-specific prompts.

```python
>>> from transformers import MvpTokenizer, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_prompt = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp-summarization")

>>> inputs = tokenizer(
...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Why You Shouldn't Quit Your Job"]

>>> generated_ids = model_with_prompt.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
["Don't do it if these are your reasons"]
```

For data-to-text generation, it is an example to use MVP and multi-task pre-trained variants.
```python
>>> from transformers import MvpTokenizerFast, MvpForConditionalGeneration

>>> tokenizer = MvpTokenizerFast.from_pretrained("RUCAIBox/mvp")
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")
>>> model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")

>>> inputs = tokenizer(
...     "Describe the following data: Iron Man | instance of | Superhero [SEP] Stan Lee | creator | Iron Man",
...     return_tensors="pt",
... )
>>> generated_ids = model.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Stan Lee created the character of Iron Man, a fictional superhero appearing in American comic']

>>> generated_ids = model_with_mtl.generate(**inputs)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
['Iron Man is a fictional superhero appearing in American comic books published by Marvel Comics.']
```

For lightweight tuning, *i.e.*, fixing the model and only tuning prompts, you can load MVP with randomly initialized prompts or with task-specific prompts. Our code also supports Prefix-tuning with BART following the [original paper](https://arxiv.org/abs/2101.00190).

```python
>>> from transformers import MvpForConditionalGeneration

>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp", use_prompt=True)
>>> # the number of trainable parameters (full tuning)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
468116832

>>> # lightweight tuning with randomly initialized prompts
>>> model.set_lightweight_tuning()
>>> # the number of trainable parameters (lightweight tuning)
>>> sum(p.numel() for p in model.parameters() if p.requires_grad)
61823328

>>> # lightweight tuning with task-specific prompts
>>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text")
>>> model.set_lightweight_tuning()
>>> # original lightweight Prefix-tuning
>>> model = MvpForConditionalGeneration.from_pretrained("facebook/bart-large", use_prompt=True)
>>> model.set_lightweight_tuning()
```

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## MvpConfig

[[autodoc]] MvpConfig

## MvpTokenizer

[[autodoc]] MvpTokenizer

## MvpTokenizerFast

[[autodoc]] MvpTokenizerFast

## MvpModel

[[autodoc]] MvpModel
    - forward

## MvpForConditionalGeneration

[[autodoc]] MvpForConditionalGeneration
    - forward

## MvpForSequenceClassification

[[autodoc]] MvpForSequenceClassification
    - forward

## MvpForQuestionAnswering

[[autodoc]] MvpForQuestionAnswering
    - forward

## MvpForCausalLM

[[autodoc]] MvpForCausalLM
    - forward
