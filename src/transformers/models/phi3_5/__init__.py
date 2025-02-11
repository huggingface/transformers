from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_torchvision_available
)


_import_structure = {
    "configuration_phi3_v": ["Phi3VConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_phi3_v"] = [
        "Phi3VForCausalLM",
        "Phi3VForSequenceClassification",
        "Phi3VForTokenClassification",
        "Phi3VModel",
        "Phi3VPreTrainedModel",
    ]
    _import_structure["processing_phi3_v"] = ["Phi3VProcessor"]
    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        _import_structure["image_processing_phi3_v"] = ["Phi3VImageProcessor"]


if TYPE_CHECKING:
    from .configuration_phi3_v import Phi3VConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_phi3_v import (
            Phi3VForCausalLM,
            Phi3VForSequenceClassification,
            Phi3VForTokenClassification,
            Phi3VModel,
            Phi3VPreTrainedModel,
        )
        from .processing_phi3_v import Phi3VProcessor
        try:
            if not is_torchvision_available():
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            pass
        else:
            from .image_processing_phi3_v import Phi3VImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

#from .configuration_phi3_v import *
#from .modeling_phi3_v import *
#from .processing_phi3_v import *
