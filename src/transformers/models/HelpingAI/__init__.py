# Made by KingNish

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from ...utils import OptionalDependencyNotAvailable


_import_structure = {"configuration_HelpingAi": ["HelpingAIConfig"]}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_HelpingAi_fast"] = ["HelpingAITokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_HelpingAi"] = [
        "HelpingAIForCausalLM",
        "HelpingAIForQuestionAnswering",
        "HelpingAIForSequenceClassification",
        "HelpingAIForTokenClassification",
        "HelpingAILayer",
        "HelpingAIModel",
        "HelpingAIPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_helpingai import HelpingAIConfig

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_helpingai_fast import HelpingAITokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_helpingai import (
            HelpingAIForCausalLM,
            HelpingAIForQuestionAnswering,
            HelpingAIForSequenceClassification,
            HelpingAIForTokenClassification,
            HelpingAILayer,
            HelpingAIModel,
            HelpingAIPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
