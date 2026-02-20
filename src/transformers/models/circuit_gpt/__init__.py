from typing import TYPE_CHECKING
from ...utils import _LazyModule

_import_structure = {
    "configuration_circuit_gpt": ["CircuitGptConfig"],
    "modeling_circuit_gpt": [
        "CircuitGptModel",
        "CircuitGptPreTrainedModel",
        "CircuitGptForCausalLM",
    ],
}

if TYPE_CHECKING:
    from .configuration_circuit_gpt import CircuitGptConfig
    from .modeling_circuit_gpt import (
        CircuitGptModel,
        CircuitGptPreTrainedModel,
        CircuitGptForCausalLM,
    )
else:
    import sys

    self = sys.modules[__name__]
    self.__class__ = _LazyModule
