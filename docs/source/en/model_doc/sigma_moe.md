<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Sigma-MoE

## Overview

TODO

## Usage tips

TODO - How to load the model?

## Resources

TODO - Talk about triton requirements?

## SigmaMoEConfiguration

[[autodoc]] SigmaMoEConfiguration

## SigmaMoETokenizer

[[autodoc]] SigmaMoETokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## SigmaMoETokenizerFast

[[autodoc]] SigmaMoETokenizerFast
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - update_post_processor
    - save_vocabulary

## SigmaMoEModel

[[autodoc]] SigmaMoEModel
    - forward

## SigmaMoEForCausalLM

[[autodoc]] SigmaMoEForCausalLM
    - forward

## SigmaMoEForSequenceClassification

[[autodoc]] SigmaMoEForSequenceClassification
    - forward

## SigmaMoEForTokenClassification

[[autodoc]] SigmaMoEForTokenClassification
    - forward