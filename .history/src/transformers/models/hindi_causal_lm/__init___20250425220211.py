

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_hindi_causal_lm": ["HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP", "HindiCausalLMConfig"],
    # Add tokenization if you have a specific file/class
    # "tokenization_hindi_causal_lm": ["HindiCausalLMTokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # Only list classes that actually exist in modeling_hindi_causal_lm.py
    _import_structure["modeling_hindi_causal_lm"] = [
        # "HindiCausalLMModel", # REMOVED - Class no longer exists in flattened structure
        "HindiCausalLMHeadModel",
    ]


if TYPE_CHECKING:
    from .configuration_hindi_causal_lm import HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig
    # from .tokenization_hindi_causal_lm import HindiCausalLMTokenizer # If exists

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # Only list classes that actually exist
        from .modeling_hindi_causal_lm import (
            # HindiCausalLMModel, # REMOVED
            HindiCausalLMHeadModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)