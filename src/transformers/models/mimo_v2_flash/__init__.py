from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_mimo_v2_flash": ["MiMoV2FlashConfig"],
    "modeling_mimo_v2_flash": [
        "MiMoV2Model",
        "MiMoV2FlashForCausalLM",
        "MiMoV2RMSNorm",
        "MiMoV2MLP",
        "MiMoV2MoEGate",
        "MiMoV2MoE",
        "MiMoV2Attention",
        "MiMoV2DecoderLayer",
        "MiMoV2FlashRotaryEmbedding",
    ],
}

if TYPE_CHECKING:
    from .configuration_mimo_v2_flash import MiMoV2FlashConfig
    from .modeling_mimo_v2_flash import (
        MiMoV2Attention,
        MiMoV2DecoderLayer,
        MiMoV2FlashForCausalLM,
        MiMoV2FlashRotaryEmbedding,
        MiMoV2MLP,
        MiMoV2Model,
        MiMoV2MoE,
        MiMoV2MoEGate,
        MiMoV2RMSNorm,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
