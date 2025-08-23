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

# FNet

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The FNet model was proposed in [FNet: Mixing Tokens with Fourier Transforms](https://huggingface.co/papers/2105.03824) by
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon. The model replaces the self-attention layer in a BERT
model with a fourier transform which returns only the real parts of the transform. The model is significantly faster
than the BERT model because it has fewer parameters and is more memory efficient. The model achieves about 92-97%
accuracy of BERT counterparts on GLUE benchmark, and trains much faster than the BERT model. The abstract from the
paper is the following:

*We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the
self-attention sublayers with simple linear transformations that "mix" input tokens. These linear mixers, along with
standard nonlinearities in feed-forward layers, prove competent at modeling semantic relationships in several text
classification tasks. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder
with a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of BERT counterparts on the GLUE
benchmark, but trains 80% faster on GPUs and 70% faster on TPUs at standard 512 input lengths. At longer input lengths,
our FNet model is significantly faster: when compared to the "efficient" Transformers on the Long Range Arena
benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all
sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint
and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models
outperform Transformer counterparts.*

This model was contributed by [gchhablani](https://huggingface.co/gchhablani). The original code can be found [here](https://github.com/google-research/google-research/tree/master/f_net).

## Usage tips

The model was trained without an attention mask as it is based on Fourier Transform. The model was trained with 
maximum sequence length 512 which includes pad tokens. Hence, it is highly recommended to use the same maximum 
sequence length for fine-tuning and inference.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## FNetConfig

[[autodoc]] FNetConfig

## FNetTokenizer

[[autodoc]] FNetTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## FNetTokenizerFast

[[autodoc]] FNetTokenizerFast

## FNetModel

[[autodoc]] FNetModel
    - forward

## FNetForPreTraining

[[autodoc]] FNetForPreTraining
    - forward

## FNetForMaskedLM

[[autodoc]] FNetForMaskedLM
    - forward

## FNetForNextSentencePrediction

[[autodoc]] FNetForNextSentencePrediction
    - forward

## FNetForSequenceClassification

[[autodoc]] FNetForSequenceClassification
    - forward

## FNetForMultipleChoice

[[autodoc]] FNetForMultipleChoice
    - forward

## FNetForTokenClassification

[[autodoc]] FNetForTokenClassification
    - forward

## FNetForQuestionAnswering

[[autodoc]] FNetForQuestionAnswering
    - forward
