# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

from ...file_utils import is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from .tokenization_lxmert import LxmertTokenizer


if is_tokenizers_available():
    from .tokenization_lxmert_fast import LxmertTokenizerFast

if is_torch_available():
    from .modeling_lxmert import (
        LxmertEncoder,
        LxmertForPreTraining,
        LxmertForQuestionAnswering,
        LxmertModel,
        LxmertPreTrainedModel,
        LxmertVisualFeatureEncoder,
        LxmertXLayer,
    )

if is_tf_available():
    from .modeling_tf_lxmert import (
        TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        TFLxmertForPreTraining,
        TFLxmertMainLayer,
        TFLxmertModel,
        TFLxmertPreTrainedModel,
        TFLxmertVisualFeatureEncoder,
    )
