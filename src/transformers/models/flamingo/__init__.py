from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_flamingo": [
        "FlamingoConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flamingo"] = [
        "FlamingoModel",
        "FlamingoPreTrainedModel",
        "FlamingoForConditionalGeneration",
    ]

if TYPE_CHECKING:
    from .configuration_flamingo import (
        FlamingoConfig,
    )

    # from .processing_flamingo import FlamingoProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flamingo import (
            FlamingoForConditionalGeneration,
            FlamingoModel,
            FlamingoPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
