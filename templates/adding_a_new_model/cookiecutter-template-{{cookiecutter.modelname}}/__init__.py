# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

{%- if cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" %}
from ...file_utils import is_tf_available, is_torch_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "PyTorch" %}
from ...file_utils import is_torch_available
{%- elif cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow" %}
from ...file_utils import is_tf_available
{% endif %}
from .configuration_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.uppercase_modelname}}_PRETRAINED_CONFIG_ARCHIVE_MAP, {{cookiecutter.camelcase_modelname}}Config
from .tokenization_{{cookiecutter.lowercase_modelname}} import {{cookiecutter.camelcase_modelname}}Tokenizer

{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "PyTorch") %}
if is_torch_available():
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
{% endif %}
{%- if (cookiecutter.generate_tensorflow_and_pytorch == "PyTorch & TensorFlow" or cookiecutter.generate_tensorflow_and_pytorch == "TensorFlow") %}
if is_tf_available():
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
{% endif %}