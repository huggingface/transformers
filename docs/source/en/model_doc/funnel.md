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

# Funnel Transformer

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=funnel">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-funnel-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/funnel-transformer-small">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>


## Overview

The Funnel Transformer model was proposed in the paper [Funnel-Transformer: Filtering out Sequential Redundancy for
Efficient Language Processing](https://arxiv.org/abs/2006.03236). It is a bidirectional transformer model, like
BERT, but with a pooling operation after each block of layers, a bit like in traditional convolutional neural networks
(CNN) in computer vision.

The abstract from the paper is the following:

*With the success of language pretraining, it is highly desirable to develop more efficient architectures of good
scalability that can exploit the abundant unlabeled data at a lower cost. To improve the efficiency, we examine the
much-overlooked redundancy in maintaining a full-length token-level presentation, especially for tasks that only
require a single-vector presentation of the sequence. With this intuition, we propose Funnel-Transformer which
gradually compresses the sequence of hidden states to a shorter one and hence reduces the computation cost. More
importantly, by re-investing the saved FLOPs from length reduction in constructing a deeper or wider model, we further
improve the model capacity. In addition, to perform token-level predictions as required by common pretraining
objectives, Funnel-Transformer is able to recover a deep representation for each token from the reduced hidden sequence
via a decoder. Empirically, with comparable or fewer FLOPs, Funnel-Transformer outperforms the standard Transformer on
a wide variety of sequence-level prediction tasks, including text classification, language understanding, and reading
comprehension.*

Tips:

- Since Funnel Transformer uses pooling, the sequence length of the hidden states changes after each block of layers. This way, their length is divided by 2, which speeds up the computation of the next hidden states.
  The base model therefore has a final sequence length that is a quarter of the original one. This model can be used
  directly for tasks that just require a sentence summary (like sequence classification or multiple choice). For other
  tasks, the full model is used; this full model has a decoder that upsamples the final hidden states to the same
  sequence length as the input.
- For tasks such as classification, this is not a problem, but for tasks like masked language modeling or token classification, we need a hidden state with the same sequence length as the original input. In those cases, the final hidden states are upsampled to the input sequence length and go through two additional layers. That's why there are two versions of each checkpoint. The version suffixed with “-base” contains only the three blocks, while the version without that suffix contains the three blocks and the upsampling head with its additional layers.
- The Funnel Transformer checkpoints are all available with a full version and a base version. The first ones should be
  used for [`FunnelModel`], [`FunnelForPreTraining`],
  [`FunnelForMaskedLM`], [`FunnelForTokenClassification`] and
  [`FunnelForQuestionAnswering`]. The second ones should be used for
  [`FunnelBaseModel`], [`FunnelForSequenceClassification`] and
  [`FunnelForMultipleChoice`].

This model was contributed by [sgugger](https://huggingface.co/sgugger). The original code can be found [here](https://github.com/laiguokun/Funnel-Transformer).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)


## FunnelConfig

[[autodoc]] FunnelConfig

## FunnelTokenizer

[[autodoc]] FunnelTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FunnelTokenizerFast

[[autodoc]] FunnelTokenizerFast

## Funnel specific outputs

[[autodoc]] models.funnel.modeling_funnel.FunnelForPreTrainingOutput

[[autodoc]] models.funnel.modeling_tf_funnel.TFFunnelForPreTrainingOutput

## FunnelBaseModel

[[autodoc]] FunnelBaseModel
    - forward

## FunnelModel

[[autodoc]] FunnelModel
    - forward

## FunnelModelForPreTraining

[[autodoc]] FunnelForPreTraining
    - forward

## FunnelForMaskedLM

[[autodoc]] FunnelForMaskedLM
    - forward

## FunnelForSequenceClassification

[[autodoc]] FunnelForSequenceClassification
    - forward

## FunnelForMultipleChoice

[[autodoc]] FunnelForMultipleChoice
    - forward

## FunnelForTokenClassification

[[autodoc]] FunnelForTokenClassification
    - forward

## FunnelForQuestionAnswering

[[autodoc]] FunnelForQuestionAnswering
    - forward

## TFFunnelBaseModel

[[autodoc]] TFFunnelBaseModel
    - call

## TFFunnelModel

[[autodoc]] TFFunnelModel
    - call

## TFFunnelModelForPreTraining

[[autodoc]] TFFunnelForPreTraining
    - call

## TFFunnelForMaskedLM

[[autodoc]] TFFunnelForMaskedLM
    - call

## TFFunnelForSequenceClassification

[[autodoc]] TFFunnelForSequenceClassification
    - call

## TFFunnelForMultipleChoice

[[autodoc]] TFFunnelForMultipleChoice
    - call

## TFFunnelForTokenClassification

[[autodoc]] TFFunnelForTokenClassification
    - call

## TFFunnelForQuestionAnswering

[[autodoc]] TFFunnelForQuestionAnswering
    - call
