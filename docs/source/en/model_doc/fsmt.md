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

# FSMT

## Overview

FSMT (FairSeq MachineTranslation) models were introduced in [Facebook FAIR's WMT19 News Translation Task Submission](https://arxiv.org/abs/1907.06616) by Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli, Sergey Edunov.

The abstract of the paper is the following:

*This paper describes Facebook FAIR's submission to the WMT19 shared news translation task. We participate in two
language pairs and four language directions, English <-> German and English <-> Russian. Following our submission from
last year, our baseline systems are large BPE-based transformer models trained with the Fairseq sequence modeling
toolkit which rely on sampled back-translations. This year we experiment with different bitext data filtering schemes,
as well as with adding filtered back-translated data. We also ensemble and fine-tune our models on domain-specific
data, then decode using noisy channel model reranking. Our submissions are ranked first in all four directions of the
human evaluation campaign. On En->De, our system significantly outperforms other systems as well as human translations.
This system improves upon our WMT'18 submission by 4.5 BLEU points.*

This model was contributed by [stas](https://huggingface.co/stas). The original code can be found
[here](https://github.com/pytorch/fairseq/tree/master/examples/wmt19).

## Implementation Notes

- FSMT uses source and target vocabulary pairs that aren't combined into one. It doesn't share embeddings tokens
  either. Its tokenizer is very similar to [`XLMTokenizer`] and the main model is derived from
  [`BartModel`].


## FSMTConfig

[[autodoc]] FSMTConfig

## FSMTTokenizer

[[autodoc]] FSMTTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FSMTModel

[[autodoc]] FSMTModel
    - forward

## FSMTForConditionalGeneration

[[autodoc]] FSMTForConditionalGeneration
    - forward
