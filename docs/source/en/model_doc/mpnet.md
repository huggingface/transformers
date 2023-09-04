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

# MPNet

## Overview

The MPNet model was proposed in [MPNet: Masked and Permuted Pre-training for Language Understanding](https://arxiv.org/abs/2004.09297) by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.

MPNet adopts a novel pre-training method, named masked and permuted language modeling, to inherit the advantages of
masked language modeling and permuted language modeling for natural language understanding.

The abstract from the paper is the following:

*BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models.
Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for
pre-training to address this problem. However, XLNet does not leverage the full position information of a sentence and
thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel
pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the
dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position
information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in
XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of
down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large
margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g.,
BERT, XLNet, RoBERTa) under the same model setting.*

Tips:

- MPNet doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. just
  separate your segments with the separation token `tokenizer.sep_token` (or `[sep]`).

The original code can be found [here](https://github.com/microsoft/MPNet).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## MPNetConfig

[[autodoc]] MPNetConfig

## MPNetTokenizer

[[autodoc]] MPNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## MPNetTokenizerFast

[[autodoc]] MPNetTokenizerFast

## MPNetModel

[[autodoc]] MPNetModel
    - forward

## MPNetForMaskedLM

[[autodoc]] MPNetForMaskedLM
    - forward

## MPNetForSequenceClassification

[[autodoc]] MPNetForSequenceClassification
    - forward

## MPNetForMultipleChoice

[[autodoc]] MPNetForMultipleChoice
    - forward

## MPNetForTokenClassification

[[autodoc]] MPNetForTokenClassification
    - forward

## MPNetForQuestionAnswering

[[autodoc]] MPNetForQuestionAnswering
    - forward

## TFMPNetModel

[[autodoc]] TFMPNetModel
    - call

## TFMPNetForMaskedLM

[[autodoc]] TFMPNetForMaskedLM
    - call

## TFMPNetForSequenceClassification

[[autodoc]] TFMPNetForSequenceClassification
    - call

## TFMPNetForMultipleChoice

[[autodoc]] TFMPNetForMultipleChoice
    - call

## TFMPNetForTokenClassification

[[autodoc]] TFMPNetForTokenClassification
    - call

## TFMPNetForQuestionAnswering

[[autodoc]] TFMPNetForQuestionAnswering
    - call
