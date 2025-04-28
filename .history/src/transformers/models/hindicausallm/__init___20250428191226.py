# flake8: noqa
from typing import TYPE_CHECKING

from ...utils import _LazyModule, OptionalDependencyNotAvailable, is_sentencepiece_available, is_tokenizers_available, is_torch_available

_import_structure = {
    "configuration_hindicausallm": ["HindiCausalLMConfig", "HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP"],
    "generation_config_hindicausallm": ["HindiCausalLMGenerationConfig"],
}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_hindicausallm"] = ["HindiCausalLMTokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_hindicausallm_fast"] = ["HindiCausalLMTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_hindicausallm"] = [
        "HindiCausalLMForCausalLM",
        "HindiCausalLMModel",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMForSequenceClassification",
    ]

if TYPE_CHECKING:
    from .configuration_hindicausallm import HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig
    from .generation_config_hindicausallm import HindiCausalLMGenerationConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hindicausallm import HindiCausalLMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_hindicausallm_fast import HindiCausalLMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hindicausallm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMForSequenceClassification,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)