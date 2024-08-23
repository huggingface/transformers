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

from ...utils import  _LazyModule, OptionalDependencyNotAvailable, is_tokenizers_available


{%- if "TensorFlow" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
from ...utils import is_tf_available


{% endif %}
{%- if "PyTorch" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
from ...utils import is_torch_available


{% endif %}
{%- if "Flax" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
from ...utils import is_flax_available


{% endif %}

_import_structure = {
    "configuration_{{cookiecutter.lowercase_modelname}}": ["{{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP", "{{cookiecutter.camelcase_modelname}}Config"],
    "tokenization_{{cookiecutter.lowercase_modelname}}": ["{{cookiecutter.camelcase_modelname}}Tokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_{{cookiecutter.lowercase_modelname}}_fast"] = ["{{cookiecutter.camelcase_modelname}}TokenizerFast"]

{%- if "PyTorch" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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


{%- if "TensorFlow" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
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
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_{{cookiecutter.lowercase_modelname}}"] = [
        "TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
        "TF{{cookiecutter.camelcase_modelname}}Model",
        "TF{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% endif %}
{% endif %}


{%- if "Flax" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_{{cookiecutter.lowercase_modelname}}"] = [
        "Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM",
        "Flax{{cookiecutter.camelcase_modelname}}ForCausalLM",
        "Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice",
        "Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
        "Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
        "Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification",
        "Flax{{cookiecutter.camelcase_modelname}}Layer",
        "Flax{{cookiecutter.camelcase_modelname}}Model",
        "Flax{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% else %}
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_{{cookiecutter.lowercase_modelname}}"] = [
        "Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
        "Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
        "Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
        "Flax{{cookiecutter.camelcase_modelname}}Model",
        "Flax{{cookiecutter.camelcase_modelname}}PreTrainedModel",
    ]
{% endif %}
{% endif %}


if TYPE_CHECKING:
    from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
    from .tokenization_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_{{cookiecutter.lowercase_modelname}}_fast import {{cookiecutter.camelcase_modelname}}TokenizerFast

{%- if "PyTorch" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
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
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
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
{%- if "TensorFlow" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
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
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_{{cookiecutter.lowercase_modelname}} import (
            TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            TF{{cookiecutter.camelcase_modelname}}Model,
            TF{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% endif %}
{% endif %}
{%- if "Flax" in cookiecutter.generate_tensorflow_pytorch_and_flax %}
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_{{cookiecutter.lowercase_modelname}} import (
            Flax{{cookiecutter.camelcase_modelname}}ForMaskedLM,
            Flax{{cookiecutter.camelcase_modelname}}ForCausalLM,
            Flax{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
            Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            Flax{{cookiecutter.camelcase_modelname}}ForTokenClassification,
            Flax{{cookiecutter.camelcase_modelname}}Layer,
            Flax{{cookiecutter.camelcase_modelname}}Model,
            Flax{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% else %}
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_{{cookiecutter.lowercase_modelname}} import (
            Flax{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            Flax{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            Flax{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            Flax{{cookiecutter.camelcase_modelname}}Model,
            Flax{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% endif %}
{% endif %}

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
