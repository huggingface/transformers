from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

import_structure = {
    "configuration_imagegpt": ["ImageGPTConfig"],
    "feature_extraction_imagegpt": ["ImageGPTFeatureExtractor"],
    "image_processing_imagegpt": ["ImageGPTImageProcessor"],
    "image_processing_imagegpt_fast": ["ImageGPTImageProcessorFast"],
    "modeling_imagegpt": [
        "ImageGPTModel",
        "ImageGPTForCausalImageModeling",
        "ImageGPTForImageClassification",
        "ImageGPTPreTrainedModel",
        "load_tf_weights_in_imagegpt",
    ],
}

if TYPE_CHECKING:
    from .configuration_imagegpt import *
    from .feature_extraction_imagegpt import *
    from .image_processing_imagegpt import *
    from .image_processing_imagegpt_fast import *
    from .modeling_imagegpt import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, import_structure, module_spec=__spec__)