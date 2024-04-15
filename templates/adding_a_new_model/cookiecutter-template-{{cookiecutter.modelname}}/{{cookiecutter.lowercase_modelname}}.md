<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# {{cookiecutter.modelname}}

## Overview

The {{cookiecutter.modelname}} model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>)  by <INSERT AUTHORS HERE>. <INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](<https://huggingface.co/<INSERT YOUR HF USERNAME HERE>). The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).

## {{cookiecutter.camelcase_modelname}}Config

[[autodoc]] {{cookiecutter.camelcase_modelname}}Config


## {{cookiecutter.camelcase_modelname}}Tokenizer

[[autodoc]] {{cookiecutter.camelcase_modelname}}Tokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## {{cookiecutter.camelcase_modelname}}TokenizerFast

[[autodoc]] {{cookiecutter.camelcase_modelname}}TokenizerFast


{% if "PyTorch" in cookiecutter.generate_tensorflow_pytorch_and_flax -%}
## {{cookiecutter.camelcase_modelname}}Model

[[autodoc]] {{cookiecutter.camelcase_modelname}}Model
    - forward

{% if cookiecutter.is_encoder_decoder_model == "False" %}
## {{cookiecutter.camelcase_modelname}}ForCausalLM

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForCausalLM
    - forward


## {{cookiecutter.camelcase_modelname}}ForMaskedLM

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForMaskedLM
    - forward


## {{cookiecutter.camelcase_modelname}}ForSequenceClassification

[[autodoc]] transformers.{{cookiecutter.camelcase_modelname}}ForSequenceClassification
    - forward

## {{cookiecutter.camelcase_modelname}}ForMultipleChoice

[[autodoc]] transformers.{{cookiecutter.camelcase_modelname}}ForMultipleChoice
    - forward


## {{cookiecutter.camelcase_modelname}}ForTokenClassification

[[autodoc]] transformers.{{cookiecutter.camelcase_modelname}}ForTokenClassification
    - forward


## {{cookiecutter.camelcase_modelname}}ForQuestionAnswering

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForQuestionAnswering
    - forward

{%- else %}
## {{cookiecutter.camelcase_modelname}}ForConditionalGeneration

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForConditionalGeneration
    - forward


## {{cookiecutter.camelcase_modelname}}ForSequenceClassification

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForSequenceClassification
    - forward


## {{cookiecutter.camelcase_modelname}}ForQuestionAnswering

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForQuestionAnswering
    - forward


## {{cookiecutter.camelcase_modelname}}ForCausalLM

[[autodoc]] {{cookiecutter.camelcase_modelname}}ForCausalLM
    - forward


{% endif -%}
{% endif -%}
{% if "TensorFlow" in cookiecutter.generate_tensorflow_pytorch_and_flax -%}

## TF{{cookiecutter.camelcase_modelname}}Model

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}Model
    - call

{% if cookiecutter.is_encoder_decoder_model == "False" %}
## TF{{cookiecutter.camelcase_modelname}}ForMaskedLM

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForMaskedLM
    - call


## TF{{cookiecutter.camelcase_modelname}}ForCausalLM

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForCausalLM
    - call


## TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification
    - call


## TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice
    - call


## TF{{cookiecutter.camelcase_modelname}}ForTokenClassification

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForTokenClassification
    - call


## TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering
    - call


{%- else %}
## TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration

[[autodoc]] TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration
    - call


{% endif -%}
{% endif -%}

{% if "Flax" in cookiecutter.generate_tensorflow_pytorch_and_flax -%}

## Flax{{cookiecutter.camelcase_modelname}}Model

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}Model
    - call

{% if cookiecutter.is_encoder_decoder_model == "False" %}
## Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForCausalLM

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForCausalLM
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering
    - call


{%- else %}
## Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering
    - call


## Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration

[[autodoc]] Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration
    - call


{% endif -%}
{% endif -%}
