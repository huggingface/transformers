## Copyright 2020 The HuggingFace Team. All rights reserved.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

## This file is made so that specific statements may be copied inside existing files. This is useful to copy
## import statements in __init__.py, or to complete model lists in the AUTO files.
##
## It is to be used as such:
## Put '# To replace in: "FILE_PATH"' in order to indicate the contents will be copied in the file at path FILE_PATH
## Put '# Below: "STATEMENT"' in order to copy the contents below **the first occurence** of that line in the file at FILE_PATH
## Put '# Replace with:' followed by the lines containing the content to define the content
## End a statement with '# End.'. If starting a new statement without redefining the FILE_PATH, it will continue pasting
## content in that file.
##
## Put '## COMMENT' to comment on the file.

# To replace in: "src/transformers/__init__.py"
# Below: "    # PyTorch models structure" if generating PyTorch
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    _import_structure["models.{{cookiecutter.lowercase_modelname}}"].extend(
        [
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
    )
{% else %}
    _import_structure["models.{{cookiecutter.lowercase_modelname}}"].extend(
        [
            "{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST",
            "{{cookiecutter.camelcase_modelname}}ForCausalLM",
            "{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
            "{{cookiecutter.camelcase_modelname}}ForQuestionAnswering",
            "{{cookiecutter.camelcase_modelname}}ForSequenceClassification",
            "{{cookiecutter.camelcase_modelname}}Model",
        ]
    )
{% endif -%}
# End.

# Below: "    # TensorFlow models structure" if generating TensorFlow
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" %}
    _import_structure["models.{{cookiecutter.lowercase_modelname}}"].extend(
        [
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
    )
{% else %}
    _import_structure["models.{{cookiecutter.lowercase_modelname}}"].extend(
        [
            "TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration",
            "TF{{cookiecutter.camelcase_modelname}}Model",
            "TF{{cookiecutter.camelcase_modelname}}PreTrainedModel",
        ]
    )
{% endif -%}
# End.

# Below: "    # Fast tokenizers"
# Replace with:
    _import_structure["models.{{cookiecutter.lowercase_modelname}}"].append("{{cookiecutter.camelcase_modelname}}TokenizerFast")
# End.

# Below: "    # Models"
# Replace with:
    "models.{{cookiecutter.lowercase_modelname}}": ["{{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP", "{{cookiecutter.camelcase_modelname}}Config", "{{cookiecutter.camelcase_modelname}}Tokenizer"],
# End.

# To replace in: "src/transformers/__init__.py"
# Below: "    if is_torch_available():" if generating PyTorch
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" %}
        from .models.{{cookiecutter.lowercase_modelname}} import (
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
        from .models.{{cookiecutter.lowercase_modelname}} import (
            {{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
            {{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            {{cookiecutter.camelcase_modelname}}ForCausalLM,
            {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
            {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
            {{cookiecutter.camelcase_modelname}}Model,
        )
{% endif -%}
# End.

# Below: "    if is_tf_available():" if generating TensorFlow
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" %}
        from .models.{{cookiecutter.lowercase_modelname}} import (
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
        from .models.{{cookiecutter.lowercase_modelname}} import (
            TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
            TF{{cookiecutter.camelcase_modelname}}Model,
            TF{{cookiecutter.camelcase_modelname}}PreTrainedModel,
        )
{% endif -%}
# End.

# Below: "    if is_tokenizers_available():"
# Replace with:
        from .models.{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}TokenizerFast
# End.

# Below: "    from .models.albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig"
# Replace with:
    from .models.{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}Tokenizer
# End.



# To replace in: "src/transformers/models/__init__.py"
# Below: "from . import ("
# Replace with:
    {{cookiecutter.lowercase_modelname}},
# End.


# To replace in: "src/transformers/models/auto/configuration_auto.py"
# Below: "# Add configs here"
# Replace with:
        ("{{cookiecutter.lowercase_modelname}}", {{cookiecutter.camelcase_modelname}}Config),
# End.

# Below: "# Add archive maps here"
# Replace with:
        {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP,
# End.

# Below: "from ..albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig",
# Replace with:
from ..{{cookiecutter.lowercase_modelname}}.configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
# End.

# Below: "# Add full (and cased) model names here"
# Replace with:
        ("{{cookiecutter.lowercase_modelname}}", "{{cookiecutter.camelcase_modelname}}"),
# End.



# To replace in: "src/transformers/models/auto/modeling_auto.py" if generating PyTorch
# Below: "from .configuration_auto import ("
# Replace with:
    {{cookiecutter.camelcase_modelname}}Config,
# End.

# Below: "# Add modeling imports here"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
from ..{{cookiecutter.lowercase_modelname}}.modeling_{{cookiecutter.lowercase_modelname}} import (
    {{cookiecutter.camelcase_modelname}}ForMaskedLM,
    {{cookiecutter.camelcase_modelname}}ForCausalLM,
    {{cookiecutter.camelcase_modelname}}ForMultipleChoice,
    {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
    {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
    {{cookiecutter.camelcase_modelname}}ForTokenClassification,
    {{cookiecutter.camelcase_modelname}}Model,
)
{% else -%}
from ..{{cookiecutter.lowercase_modelname}}.modeling_{{cookiecutter.lowercase_modelname}} import (
    {{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
    {{cookiecutter.camelcase_modelname}}ForCausalLM,
    {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
    {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
    {{cookiecutter.camelcase_modelname}}Model,
)
{% endif -%}
# End.

# Below: "# Base model mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}Model),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMaskedLM),
{% else %}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForConditionalGeneration),
{% endif -%}
# End.

# Below: "# Model for Causal LM mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForCausalLM),
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMaskedLM),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Sequence Classification mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForSequenceClassification),
# End.

# Below: "# Model for Question Answering mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForQuestionAnswering),
# End.

# Below: "# Model for Token Classification mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForTokenClassification),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMultipleChoice),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Seq2Seq Causal LM mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
{% else %}
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForConditionalGeneration),
{% endif -%}
# End.

# To replace in: "src/transformers/models/auto/modeling_tf_auto.py" if generating TensorFlow
# Below: "from .configuration_auto import ("
# Replace with:
    {{cookiecutter.camelcase_modelname}}Config,
# End.

# Below: "# Add modeling imports here"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
from ..{{cookiecutter.lowercase_modelname}}.modeling_tf_{{cookiecutter.lowercase_modelname}} import (
    TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
    TF{{cookiecutter.camelcase_modelname}}ForCausalLM,
    TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
    TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
    TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
    TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
    TF{{cookiecutter.camelcase_modelname}}Model,
)
{% else -%}
from ..{{cookiecutter.lowercase_modelname}}.modeling_tf_{{cookiecutter.lowercase_modelname}} import (
    TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration,
    TF{{cookiecutter.camelcase_modelname}}Model,
)
{% endif -%}
# End.

# Below: "# Base model mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}Model),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMaskedLM),
{% else %}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration),
{% endif -%}
# End.

# Below: "# Model for Causal LM mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForCausalLM),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMaskedLM),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Sequence Classification mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Question Answering mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Token Classification mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForTokenClassification),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice),
{% else -%}
{% endif -%}
# End.

# Below: "# Model for Seq2Seq Causal LM mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
{% else %}
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForConditionalGeneration),
{% endif -%}
# End.

# To replace in: "utils/check_repo.py" if generating PyTorch

# Below: "models to ignore for model xxx mapping"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
{% else -%}
    "{{cookiecutter.camelcase_modelname}}Encoder",
    "{{cookiecutter.camelcase_modelname}}Decoder",
    "{{cookiecutter.camelcase_modelname}}DecoderWrapper",
{% endif -%}
# End.

# Below: "models to ignore for not tested"
# Replace with:
{% if cookiecutter.is_encoder_decoder_model == "False" -%}
{% else -%}
    "{{cookiecutter.camelcase_modelname}}Encoder",  # Building part of bigger (tested) model.
    "{{cookiecutter.camelcase_modelname}}Decoder",  # Building part of bigger (tested) model.
    "{{cookiecutter.camelcase_modelname}}DecoderWrapper", # Building part of bigger (tested) model.
{% endif -%}
# End.
