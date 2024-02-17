from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)


_import_structure = {
    "configuration_hiera": [
        "HIREA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "HireaConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["hirea"] = [
        "HIREA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Hirea",
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