<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# GeoLM

## Overview

<!-- 
TODO: 
The GeoLM model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE> -->

GeoLM is a language model built upon BERT to enhance geospatial understanding for natural language corpus. It is pretrained on world-wide OpenStreetMap (OSM), WikiData and Wikipedia data using masked language modeling (MLM) and contrastive learning. GeoLM can be adapted to various downstream tasks, such as [toponym recognition](https://huggingface.co/zekun-li/geolm-base-toponym-recognition) and toponym linking. 


<!-- This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>). -->



## GeoLMConfig

[[autodoc]] GeoLMConfig


## GeoLMTokenizer

[[autodoc]] GeoLMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## GeoLMTokenizerFast

[[autodoc]] GeoLMTokenizerFast


## GeoLMModel

[[autodoc]] GeoLMModel
    - forward


## GeoLMForCausalLM

[[autodoc]] GeoLMForCausalLM
    - forward


## GeoLMForMaskedLM

[[autodoc]] GeoLMForMaskedLM
    - forward

## GeoLMForTokenClassification

[[autodoc]] transformers.GeoLMForTokenClassification
    - forward

