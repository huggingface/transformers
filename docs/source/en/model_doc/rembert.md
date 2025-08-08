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

# RemBERT

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

## Overview

The RemBERT model was proposed in [Rethinking Embedding Coupling in Pre-trained Language Models](https://huggingface.co/papers/2010.12821) by Hyung Won Chung, Thibault Févry, Henry Tsai, Melvin Johnson, Sebastian Ruder.

The abstract from the paper is the following:

*We re-evaluate the standard practice of sharing weights between input and output embeddings in state-of-the-art
pre-trained language models. We show that decoupled embeddings provide increased modeling flexibility, allowing us to
significantly improve the efficiency of parameter allocation in the input embedding of multilingual models. By
reallocating the input embedding parameters in the Transformer layers, we achieve dramatically better performance on
standard natural language understanding tasks with the same number of parameters during fine-tuning. We also show that
allocating additional capacity to the output embedding provides benefits to the model that persist through the
fine-tuning stage even though the output embedding is discarded after pre-training. Our analysis shows that larger
output embeddings prevent the model's last layers from overspecializing to the pre-training task and encourage
Transformer representations to be more general and more transferable to other tasks and languages. Harnessing these
findings, we are able to train models that achieve strong performance on the XTREME benchmark without increasing the
number of parameters at the fine-tuning stage.*

## Usage tips

For fine-tuning, RemBERT can be thought of as a bigger version of mBERT with an ALBERT-like factorization of the
embedding layer. The embeddings are not tied in pre-training, in contrast with BERT, which enables smaller input
embeddings (preserved during fine-tuning) and bigger output embeddings (discarded at fine-tuning). The tokenizer is
also similar to the Albert one rather than the BERT one.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## RemBertConfig

[[autodoc]] RemBertConfig

## RemBertTokenizer

[[autodoc]] RemBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RemBertTokenizerFast

[[autodoc]] RemBertTokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

<frameworkcontent>
<pt>

## RemBertModel

[[autodoc]] RemBertModel
    - forward

## RemBertForCausalLM

[[autodoc]] RemBertForCausalLM
    - forward

## RemBertForMaskedLM

[[autodoc]] RemBertForMaskedLM
    - forward

## RemBertForSequenceClassification

[[autodoc]] RemBertForSequenceClassification
    - forward

## RemBertForMultipleChoice

[[autodoc]] RemBertForMultipleChoice
    - forward

## RemBertForTokenClassification

[[autodoc]] RemBertForTokenClassification
    - forward

## RemBertForQuestionAnswering

[[autodoc]] RemBertForQuestionAnswering
    - forward

</pt>
<tf>

## TFRemBertModel

[[autodoc]] TFRemBertModel
    - call

## TFRemBertForMaskedLM

[[autodoc]] TFRemBertForMaskedLM
    - call

## TFRemBertForCausalLM

[[autodoc]] TFRemBertForCausalLM
    - call

## TFRemBertForSequenceClassification

[[autodoc]] TFRemBertForSequenceClassification
    - call

## TFRemBertForMultipleChoice

[[autodoc]] TFRemBertForMultipleChoice
    - call

## TFRemBertForTokenClassification

[[autodoc]] TFRemBertForTokenClassification
    - call

## TFRemBertForQuestionAnswering

[[autodoc]] TFRemBertForQuestionAnswering
    - call

</tf>
</frameworkcontent>
