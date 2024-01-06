<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# NarrowBERT

## Overview

The NarrowBERT model was proposed in [NarrowBERT: Accelerating Masked Language Model Pretraining and Inference
](https://arxiv.org/abs/2301.04761)  by Haoxin Li, Phillip Keung, Daniel Cheng, Jungo Kasai, Noah A. Smith. A modified transformer encoder that increases the throughput for masked language model pretraining by more than 2x and is also comparable to standard BERT performance. 

**This implementation is the SparseQueries variant.** 

The abstract from the paper is the following:

*Large-scale language model pretraining is a very successful form of self-supervised learning in natural language processing, but it is increasingly expensive to perform as the models and pretraining corpora have become larger over time. We propose NarrowBERT, a modified transformer encoder that increases the throughput for masked language model pretraining by more than 2×. NarrowBERT sparsifies the transformer model such that the self-attention queries and feedforward layers only operate on the masked tokens of each sentence during pretraining, rather than all of the tokens as with the usual transformer encoder. We also show that NarrowBERT increases the throughput at inference time by as much as 3.5× with minimal (or no) performance degradation on sentence encoding tasks like MNLI. Finally, we examine the performance of NarrowBERT on the IMDB and Amazon reviews classification and CoNLL NER tasks and show that it is also comparable to standard BERT performance.*

Tips:

- We have released two SparseQueries models [here](https://huggingface.co/models?filter=narrow_bert), including MLM pretrained model and MNLI finetuned model. 
- `NarrowBertModel` speedup depends on narrow_mask. Gain more efficiency by masking out unnecessary outputs. 

This model was contributed by [lihaoxin2020](https://huggingface.co/lihaoxin2020). The original code can be found [here](https://github.com/lihaoxin2020/narrowbert).

## NarrowBertConfig

[[autodoc]] NarrowBertConfig


## NarrowBertTokenizer

[[autodoc]] NarrowBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## NarrowBertTokenizerFast

[[autodoc]] NarrowBertTokenizerFast


## NarrowBertModel

[[autodoc]] NarrowBertModel
    - forward


## NarrowBertForMaskedLM

[[autodoc]] NarrowBertForMaskedLM
    - forward


## NarrowBertForSequenceClassification

[[autodoc]] transformers.NarrowBertForSequenceClassification
    - forward

## NarrowBertForMultipleChoice

[[autodoc]] transformers.NarrowBertForMultipleChoice
    - forward


## NarrowBertForTokenClassification

[[autodoc]] transformers.NarrowBertForTokenClassification
    - forward
