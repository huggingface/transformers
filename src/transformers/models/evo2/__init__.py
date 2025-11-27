"""Evo2 model, tokenizer, and configuration."""

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

_import_structure = {
    "configuration_evo2": ["Evo2Config"],
    "tokenization_evo2": ["Evo2Tokenizer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_evo2"] = [
        "Evo2ForCausalLM",
        "Evo2Model",
        "Evo2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_evo2 import Evo2Config
    from .tokenization_evo2 import Evo2Tokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_evo2 import Evo2ForCausalLM, Evo2Model, Evo2PreTrainedModel
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
