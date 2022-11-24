from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_vivit": ["VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViViTConfig"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vivit"] = [
        "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ViViTModel",
        "ViViTPreTrainedModel",
        "ViViTForVideoClassification",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_vivit"] = ["ViViTFeatureExtractor"]
    _import_structure["image_processing_vivit"] = ["ViViTImageProcessor"]

if TYPE_CHECKING:
    from .configuration_vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViViTConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modelling_vivit import (
            VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
            ViViTModel,
            ViViTForVideoClassification,
            ViViTPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vivit import ViViTFeatureExtractor
        from .image_processing_vivit import ViViTImageProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
