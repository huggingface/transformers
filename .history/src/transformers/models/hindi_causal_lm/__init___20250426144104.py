# coding=utf-8
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_torch_available,
)


_import_structure = {
    "configuration_hindi_causal_lm": ["HindiCausalLMConfig"],
}

# Always expose HindiCausalLMTokenizer as an attribute, real or dummy
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    HindiCausalLMTokenizer = HindiCausalLMTokenizer
else:
    _import_structure["tokenization_hindi_causal_lm"] = ["HindiCausalLMTokenizer"]

# Always expose HindiCausalLMModel, HindiCausalLMForCausalLM, HindiCausalLMPreTrainedModel as attributes, real or dummy
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from .dummy_pt_objects import (
        HindiCausalLMForCausalLM,
        HindiCausalLMModel,
        HindiCausalLMPreTrainedModel,
    )
    HindiCausalLMHeadModel = HindiCausalLMForCausalLM
    HindiCausalLMModel = HindiCausalLMModel
    HindiCausalLMForCausalLM = HindiCausalLMForCausalLM
    HindiCausalLMPreTrainedModel = HindiCausalLMPreTrainedModel
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMForCausalLM",
        "HindiCausalLMPreTrainedModel",
        "HindiCausalLMHeadModel",
    ]

if TYPE_CHECKING:

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_tokenizer_objects import HindiCausalLMTokenizer
    else:
        from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from .dummy_pt_objects import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )
        HindiCausalLMHeadModel = HindiCausalLMForCausalLM
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMForCausalLM,
            HindiCausalLMModel,
            HindiCausalLMPreTrainedModel,
        )

else:
    import sys
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
