<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# SeamlessM4T

## Overview

The SeamlessM4T model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>)  by <INSERT AUTHORS HERE>. <INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](<https://huggingface.co/<INSERT YOUR HF USERNAME HERE>). The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

## SeamlessM4TModel

[[autodoc]] SeamlessM4TModel
    - generate


## SeamlessM4TForTextToSpeech

[[autodoc]] SeamlessM4TForTextToSpeech
    - generate


## SeamlessM4TForSpeechToSpeech

[[autodoc]] SeamlessM4TForSpeechToSpeech
    - generate


## SeamlessM4TForTextToText

[[autodoc]] transformers.SeamlessM4TForTextToText
    - forward
    - generate

## SeamlessM4TForSpeechToText

[[autodoc]] transformers.SeamlessM4TForSpeechToText
    - forward
    - generate

## SeamlessM4TConfig

[[autodoc]] SeamlessM4TConfig


## SeamlessM4TTokenizer

[[autodoc]] SeamlessM4TTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## SeamlessM4TTokenizerFast

[[autodoc]] SeamlessM4TTokenizerFast


## SeamlessM4TFeatureExtractor

[[autodoc]] SeamlessM4TTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

