# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

{%- if cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" %}
from ...file_utils import _BaseLazyModule, is_tf_available, is_torch_available, is_tokenizers_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "PyTorch" %}
from ...file_utils import _BaseLazyModule, is_torch_available, is_tokenizers_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow" %}
from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available
{% endif %}
_import_structure = {
    "configuration_{{cookiecutter.lowercase_modelname}}": ["{{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP", "{{cookiecutter.camelcase_modelname}}Config"],
    "tokenization_{{cookiecutter.lowercase_modelname}}": ["{{cookiecutter.camelcase_modelname}}Tokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_{{cookiecutter.lowercase_modelname}}_fast"] = ["{{cookiecutter.camelcase_modelname}}TokenizerFast"]

{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "PyTorch") %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
if is_torch_available():
    _import_structure["modeling_{{cookiecutter.lowercase_modelname}}"] = [
        "{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST",
        "{{cookiecutter.camelcase_modelname}}ForMaskedLM",
        "{{cookiecutter.camelcase_modelname}}ForCausalLM",
        "{{cookiecutter.camelcase_modelname}}ForMultipleChoice",
        "{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
        "{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
        "{{cookiecutter.camelcase_modelname}}ForTokenClassification",
        "{{cookiecutter.camelcase_modelname}}Layer",
        "{{cookiecutter.camelcase_modelname}}Model",
        "{{cookiecutter.camelcase_modelname}}PreTrainedModel",
        "load_tf_weights_in_{{cookiecutter.lowercase_modelname}}",
    ]
{% else %}
if is_torch_available():
    _import_structure["modeling_{{cookiecutter.lowercase_modelname}}"] = [
        "{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST",
        "{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
        "{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
        "{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
        "{{cookiecutter.camelcase_modelname}}ForCausalLM",
        "{{cookiecutter.camelcase_modelname}}Model",
        "{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% endif %}
{% endif %}
{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow") %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
if is_tf_available():
    _import_structure["modeling_tf_{{cookiecutter.lowercase_modelname}}"] = [
        "TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF{{cookiecutter.camelcase_modelname}}ForMaskedLM",
        "TF{{cookiecutter.camelcase_modelname}}ForCausalLM",
        "TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice",
        "TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
        "TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
        "TF{{cookiecutter.camelcase_modelname}}ForTokenClassification",
        "TF{{cookiecutter.camelcase_modelname}}Layer",
        "TF{{cookiecutter.camelcase_modelname}}Model",
        "TF{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% else %}
if is_tf_available():
    _import_structure["modeling_tf_{{cookiecutter.lowercase_modelname}}"] = [
        "TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
        "TF{{cookiecutter.camelcase_modelname}}Model",
        "TF{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% endif %}
{% endif %}


if TYPE_CHECKING:
    from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
    from .tokenization_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Tokenizer

    if is_tokenizers_available():
        from .tokenization_{{cookiecutter.lowercase_modelname}}_fast import {{cookiecutter.camelcase_modelname}}TokenizerFast

{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "PyTorch") %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    if is_torch_available():
        from .modeling_{{cookiecutter.lowercase_modelname}} import (
            {{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
            {{cookiecutter.camelcase_modelname}}ForMaskedLM,
            {{cookiecutter.camelcase_modelname}}ForCausalLM,
            {{cookiecutter.camelcase_modelname}}ForMultipleChoice,
            {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            {{cookiecutter.camelcase_modelname}}ForTokenClassification,
            {{cookiecutter.camelcase_modelname}}Layer,
            {{cookiecutter.camelcase_modelname}}Model,
            {{cookiecutter.camelcase_modelname}}PreTrainedModel,
            load_tf_weights_in_{{cookiecutter.lowercase_modelname}},
        )
{% else %}
    if is_torch_available():
        from .modeling_{{cookiecutter.lowercase_modelname}} import (
            {{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
            {{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            {{cookiecutter.camelcase_modelname}}ForCausalLM,
            {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            {{cookiecutter.camelcase_modelname}}Model,
            {{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% endif %}
{% endif %}
{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow") %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    if is_tf_available():
        from .modeling_tf_{{cookiecutter.lowercase_modelname}} import (
            TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
            TF{{cookiecutter.camelcase_modelname}}ForCausalLM,
            TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
            TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
            TF{{cookiecutter.camelcase_modelname}}Layer,
            TF{{cookiecutter.camelcase_modelname}}Model,
            TF{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% else %}
    if is_tf_available():
        from .modeling_tf_{{cookiecutter.lowercase_modelname}} import (
            TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            TF{{cookiecutter.camelcase_modelname}}Model,
            TF{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% endif %}
{% endif %}
else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
