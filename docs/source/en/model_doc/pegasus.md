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

# Pegasus

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=pegasus">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-pegasus-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/pegasus_paraphrase">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title)
and assign @patrickvonplaten.


## Overview

The Pegasus model was proposed in [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf) by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.

According to the abstract,

- Pegasus' pretraining task is intentionally similar to summarization: important sentences are removed/masked from an
  input document and are generated together as one output sequence from the remaining sentences, similar to an
  extractive summary.
- Pegasus achieves SOTA summarization performance on all 12 downstream tasks, as measured by ROUGE and human eval.

This model was contributed by [sshleifer](https://huggingface.co/sshleifer). The Authors' code can be found [here](https://github.com/google-research/pegasus).

Tips:

- Sequence-to-sequence model with the same encoder-decoder model architecture as BART. Pegasus is pre-trained jointly on two self-supervised objective functions: Masked Language Modeling (MLM) and a novel summarization specific pretraining objective, called Gap Sentence Generation (GSG).

  * MLM: encoder input tokens are randomly replaced by a mask tokens and have to be predicted by the encoder (like in BERT)
  * GSG: whole encoder input sentences are replaced by a second mask token and fed to the decoder, but which has a causal mask to hide the future words like a regular auto-regressive transformer decoder.

## Checkpoints

All the [checkpoints](https://huggingface.co/models?search=pegasus) are fine-tuned for summarization, besides
*pegasus-large*, whence the other checkpoints are fine-tuned:

- Each checkpoint is 2.2 GB on disk and 568M parameters.
- FP16 is not supported (help/ideas on this appreciated!).
- Summarizing xsum in fp32 takes about 400ms/sample, with default parameters on a v100 GPU.
- Full replication results and correctly pre-processed data can be found in this [Issue](https://github.com/huggingface/transformers/issues/6844#issue-689259666).
- [Distilled checkpoints](https://huggingface.co/models?search=distill-pegasus) are described in this [paper](https://arxiv.org/abs/2010.13002).

### Examples

- [Script](https://github.com/huggingface/transformers/tree/main/examples/research_projects/seq2seq-distillation/finetune_pegasus_xsum.sh) to fine-tune pegasus
  on the XSUM dataset. Data download instructions at [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).
- FP16 is not supported (help/ideas on this appreciated!).
- The adafactor optimizer is recommended for pegasus fine-tuning.


## Implementation Notes

- All models are transformer encoder-decoders with 16 layers in each component.
- The implementation is completely inherited from [`BartForConditionalGeneration`]
- Some key configuration differences:

  - static, sinusoidal position embeddings
  - the model starts generating with pad_token_id (which has 0 token_embedding) as the prefix.
  - more beams are used (`num_beams=8`)
- All pretrained pegasus checkpoints are the same besides three attributes: `tokenizer.model_max_length` (maximum
  input size), `max_length` (the maximum number of tokens to generate) and `length_penalty`.
- The code to convert checkpoints trained in the author's [repo](https://github.com/google-research/pegasus) can be
  found in `convert_pegasus_tf_to_pytorch.py`.


## Usage Example

```python
>>> from transformers import PegasusForConditionalGeneration, PegasusTokenizer
>>> import torch

>>> src_text = [
...     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
... ]

... model_name = "google/pegasus-xsum"
... device = "cuda" if torch.cuda.is_available() else "cpu"
... tokenizer = PegasusTokenizer.from_pretrained(model_name)
... model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
... batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
... translated = model.generate(**batch)
... tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
... assert (
...     tgt_text[0]
...     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
... )
```

## Documentation resources

- [Causal language modeling task guide](../tasks/language_modeling)
- [Translation task guide](../tasks/translation)
- [Summarization task guide](../tasks/summarization)

## PegasusConfig

[[autodoc]] PegasusConfig

## PegasusTokenizer

warning: `add_tokens` does not work at the moment.

[[autodoc]] PegasusTokenizer

## PegasusTokenizerFast

[[autodoc]] PegasusTokenizerFast

## PegasusModel

[[autodoc]] PegasusModel
    - forward

## PegasusForConditionalGeneration

[[autodoc]] PegasusForConditionalGeneration
    - forward

## PegasusForCausalLM

[[autodoc]] PegasusForCausalLM
    - forward

## TFPegasusModel

[[autodoc]] TFPegasusModel
    - call

## TFPegasusForConditionalGeneration

[[autodoc]] TFPegasusForConditionalGeneration
    - call

## FlaxPegasusModel

[[autodoc]] FlaxPegasusModel
    - __call__
    - encode
    - decode

## FlaxPegasusForConditionalGeneration

[[autodoc]] FlaxPegasusForConditionalGeneration
    - __call__
    - encode
    - decode
