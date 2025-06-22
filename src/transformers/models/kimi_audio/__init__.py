from typing import TYPE_CHECKING
from ...utils import _LazyModule

_import_structure = {
    "configuration_kimi_audio": ["KimiAudioConfig"],
    "modeling_kimi_audio": ["KimiAudioForCausalLM"],   
    "processing_kimi_audio": ["KimiAudioProcessor"],  
}

if TYPE_CHECKING:
    from .configuration_kimi_audio import KimiAudioConfig
    from .modeling_kimi_audio import KimiAudioForCausalLM
    from .processing_kimi_audio import KimiAudioProcessor
else:
    import sys
    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )
