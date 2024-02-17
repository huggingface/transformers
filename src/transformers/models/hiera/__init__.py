from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_hiera": [
        "HIERA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "HieraConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["hiera_model"] = [
        "HIERA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "HieraModel",
        "Head",
        "HieraBlock",
        "MaskUnitAttention"
        ""
    ]


if TYPE_CHECKING:
    from .configuration_hiera import (
        HIERA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        HieraConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .hiera_model import (
            HIERA_PRETRAINED_MODEL_ARCHIVE_LIST,
            HieraModel,
            Head,
            HieraBlock,
            MaskUnitAttention,
            
        )
        from .hiera_image_processor import (
            HieraImageProcessor
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)