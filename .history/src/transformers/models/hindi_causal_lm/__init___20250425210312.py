# In transformers/models/hindi_causal_lm/__init__.py

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_hindi_causal_lm": ["HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP", "HindiCausalLMConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_hindi_causal_lm"] = [
        "HindiCausalLMModel",
        "HindiCausalLMHeadModel",
     
    ]


if TYPE_CHECKING:
    from .configuration_hindi_causal_lm import HINDI_CAUSAL_LM_PRETRAINED_CONFIG_ARCHIVE_MAP, HindiCausalLMConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_hindi_causal_lm import (
            HindiCausalLMModel,
            HindiCausalLMHeadModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)