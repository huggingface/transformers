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

# BORT

<Tip warning={true}>

This model is in maintenance mode only, so we won't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.30.0.
You can do so by running the following command: `pip install -U transformers==4.30.0`.

</Tip>

## Overview

The BORT model was proposed in [Optimal Subarchitecture Extraction for BERT](https://arxiv.org/abs/2010.10499) by
Adrian de Wynter and Daniel J. Perry. It is an optimal subset of architectural parameters for the BERT, which the
authors refer to as "Bort".

The abstract from the paper is the following:

*We extract an optimal subset of architectural parameters for the BERT architecture from Devlin et al. (2018) by
applying recent breakthroughs in algorithms for neural architecture search. This optimal subset, which we refer to as
"Bort", is demonstrably smaller, having an effective (that is, not counting the embedding layer) size of 5.5% the
original BERT-large architecture, and 16% of the net size. Bort is also able to be pretrained in 288 GPU hours, which
is 1.2% of the time required to pretrain the highest-performing BERT parametric architectural variant, RoBERTa-large
(Liu et al., 2019), and about 33% of that of the world-record, in GPU hours, required to train BERT-large on the same
hardware. It is also 7.9x faster on a CPU, as well as being better performing than other compressed variants of the
architecture, and some of the non-compressed variants: it obtains performance improvements of between 0.3% and 31%,
absolute, with respect to BERT-large, on multiple public natural language understanding (NLU) benchmarks.*

Tips:

- BORT's model architecture is based on BERT, so one can refer to [BERT's documentation page](bert) for the
  model's API as well as usage examples.
- BORT uses the RoBERTa tokenizer instead of the BERT tokenizer, so one can refer to [RoBERTa's documentation page](roberta) for the tokenizer's API as well as usage examples.
- BORT requires a specific fine-tuning algorithm, called [Agora](https://adewynter.github.io/notes/bort_algorithms_and_applications.html#fine-tuning-with-algebraic-topology) ,
  that is sadly not open-sourced yet. It would be very useful for the community, if someone tries to implement the
  algorithm to make BORT fine-tuning work.

This model was contributed by [stefan-it](https://huggingface.co/stefan-it). The original code can be found [here](https://github.com/alexa/bort/).
