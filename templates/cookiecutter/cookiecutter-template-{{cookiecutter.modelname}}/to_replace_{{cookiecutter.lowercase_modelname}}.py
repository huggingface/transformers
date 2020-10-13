# To replace in: "src/transformers/__init__.py"
# Below: "if is_torch_available():"
# Replace with:
    from .modeling_{{cookiecutter.lowercase_modelname}} import (
        {{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
        {{cookiecutter.camelcase_modelname}}ForMaskedLM,
        {{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        {{cookiecutter.camelcase_modelname}}ForTokenClassification,
        {{cookiecutter.camelcase_modelname}}Layer,
        {{cookiecutter.camelcase_modelname}}Model,
        {{cookiecutter.camelcase_modelname}}PreTrainedModel,
        load_tf_weights_in_{{cookiecutter.lowercase_modelname}},
    )
# End.

# Below: "if is_tf_available():"
# Replace with:
    from .modeling_tf_{{cookiecutter.lowercase_modelname}} import (
        TF_{{cookiecutter.uppercase_modelname}}_PRETRAINED_MODEL_ARCHIVE_LIST,
        TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
        TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
        TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
        TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
        TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
        TF{{cookiecutter.camelcase_modelname}}Layer,
        TF{{cookiecutter.camelcase_modelname}}Model,
        TF{{cookiecutter.camelcase_modelname}}PreTrainedModel,
    )
# End.


# Below: "from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig"
# Replace with:
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
# End.



# To replace in: "src/transformers/configuration_auto.py"
# Below: "# Add configs here"
# Replace with:
        ("{{cookiecutter.lowercase_modelname}}", {{cookiecutter.camelcase_modelname}}Config),
# End.

# Below: "# Add archive maps here"
# Replace with:
        {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP,
# End.

# Below: "from .configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig",
# Replace with:
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
# End.

# Below: "# Add cased model names here"
# Replace with:
        ("{{cookiecutter.lowercase_modelname}}", "{{cookiecutter.camelcase_modelname}}"),
# End.



# To replace in: "src/transformers/modeling_auto.py"
# Below: "from .configuration_auto import ("
# Replace with:
    {{cookiecutter.camelcase_modelname}}Config,
# End.

# Below: "# Add modeling imports here"
# Replace with:

from .modeling_{{cookiecutter.lowercase_modelname}} import (
    {{cookiecutter.camelcase_modelname}}ForMaskedLM,
    {{cookiecutter.camelcase_modelname}}ForMultipleChoice,
    {{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
    {{cookiecutter.camelcase_modelname}}ForSequenceClassification,
    {{cookiecutter.camelcase_modelname}}ForTokenClassification,
    {{cookiecutter.camelcase_modelname}}Model,
)
# End.

# Below: "# Base model mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}Model),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMaskedLM),
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMaskedLM),
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
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForTokenClassification),
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, {{cookiecutter.camelcase_modelname}}ForMultipleChoice),
# End.



# To replace in: "src/transformers/modeling_tf_auto.py"
# Below: "from .configuration_auto import ("
# Replace with:
    {{cookiecutter.camelcase_modelname}}Config,
# End.

# Below: "# Add modeling imports here"
# Replace with:

from .modeling_tf_{{cookiecutter.lowercase_modelname}} import (
    TF{{cookiecutter.camelcase_modelname}}ForMaskedLM,
    TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice,
    TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering,
    TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification,
    TF{{cookiecutter.camelcase_modelname}}ForTokenClassification,
    TF{{cookiecutter.camelcase_modelname}}Model,
)
# End.

# Below: "# Base model mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}Model),
# End.

# Below: "# Model with LM heads mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMaskedLM),
# End.

# Below: "# Model for Masked LM mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMaskedLM),
# End.

# Below: "# Model for Sequence Classification mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForSequenceClassification),
# End.

# Below: "# Model for Question Answering mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForQuestionAnswering),
# End.

# Below: "# Model for Token Classification mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForTokenClassification),
# End.

# Below: "# Model for Multiple Choice mapping"
# Replace with:
        ({{cookiecutter.camelcase_modelname}}Config, TF{{cookiecutter.camelcase_modelname}}ForMultipleChoice),
# End.



