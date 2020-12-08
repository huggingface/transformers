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

{%- if cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" %}
from ...file_utils import is_tf_available, is_torch_available, is_tokenizers_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "PyTorch" %}
from ...file_utils import is_torch_available, is_tokenizers_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow" %}
from ...file_utils import is_tf_available, is_tokenizers_available
{% endif %}
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
from .tokenization_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Tokenizer

if is_tokenizers_available():
    from .tokenization_{{cookiecutter.lowercase_modelname}}_fast import {{cookiecutter.camelcase_modelname}}TokenizerFast

{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "PyTorch") %}
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
{% endif %}
{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow") %}
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
{% endif %}