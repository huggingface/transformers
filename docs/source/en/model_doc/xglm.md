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

# XGLM

## Overview

The XGLM model was proposed in [Few-shot Learning with Multilingual Language Models](https://arxiv.org/abs/2112.10668)
by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, 
Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, 
Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.

The abstract from the paper is the following:

*Large-scale autoregressive language models such as GPT-3 are few-shot learners that can perform a wide range of language 
tasks without fine-tuning. While these models are known to be able to jointly represent many different languages, 
their training data is dominated by English, potentially limiting their cross-lingual generalization. 
In this work, we train multilingual autoregressive language models on a balanced corpus covering a diverse set of languages, 
and study their few- and zero-shot learning capabilities in a wide range of tasks. Our largest model with 7.5 billion parameters 
sets new state of the art in few-shot learning in more than 20 representative languages, outperforming GPT-3 of comparable size 
in multilingual commonsense reasoning (with +7.4% absolute accuracy improvement in 0-shot settings and +9.4% in 4-shot settings) 
and natural language inference (+5.4% in each of 0-shot and 4-shot settings). On the FLORES-101 machine translation benchmark, 
our model outperforms GPT-3 on 171 out of 182 translation directions with 32 training examples, while surpassing the 
official supervised baseline in 45 directions. We present a detailed analysis of where the model succeeds and fails, 
showing in particular that it enables cross-lingual in-context learning on some tasks, while there is still room for improvement 
on surface form robustness and adaptation to tasks that do not have a natural cloze form. Finally, we evaluate our models 
in social value tasks such as hate speech detection in five languages and find it has limitations similar to comparable sized GPT-3 models.*


This model was contributed by [Suraj](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/pytorch/fairseq/tree/main/examples/xglm).

## Documentation resources

- [Causal language modeling task guide](../tasks/language_modeling)

## XGLMConfig

[[autodoc]] XGLMConfig

## XGLMTokenizer

[[autodoc]] XGLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## XGLMTokenizerFast

[[autodoc]] XGLMTokenizerFast

## XGLMModel

[[autodoc]] XGLMModel
    - forward

## XGLMForCausalLM

[[autodoc]] XGLMForCausalLM
    - forward

## TFXGLMModel

[[autodoc]] TFXGLMModel
    - call

## TFXGLMForCausalLM

[[autodoc]] TFXGLMForCausalLM
    - call

## FlaxXGLMModel

[[autodoc]] FlaxXGLMModel
    - __call__

## FlaxXGLMForCausalLM

[[autodoc]] FlaxXGLMForCausalLM
    - __call__