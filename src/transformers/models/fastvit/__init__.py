from typing import TYPE_CHECKING

# rely on isort to merge the imports
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {"configuration_fastvit": ["FASTVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "FastViTConfig"]}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_fastvit"] = [
        "FASTVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FastViTForImageClassification",
        "FastViTForMaskedImageModeling",
        "FastViTModel",
        "FastViTPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_fastvit import FASTVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, FastViTConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_fastvit import (
            FASTVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
            FastViTForImageClassification,
            FastViTForMaskedImageModeling,
            FastViTModel,
            FastViTPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)