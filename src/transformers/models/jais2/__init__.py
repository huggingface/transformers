from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_jais2": ["Jais2Config"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_jais2"] = [
        "Jais2ForCausalLM",
        "Jais2ForQuestionAnswering",
        "Jais2ForSequenceClassification",
        "Jais2ForTokenClassification",
        "Jais2Model",
        "Jais2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_jais2 import Jais2Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_jais2 import (
            Jais2ForCausalLM,
            Jais2ForQuestionAnswering,
            Jais2ForSequenceClassification,
            Jais2ForTokenClassification,
            Jais2Model,
            Jais2PreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
