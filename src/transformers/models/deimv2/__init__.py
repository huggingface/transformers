from typing import Dict, List

# Lazy import structure used across Transformers
from ...utils import _LazyModule, OptionalDependencyNotAvailable
import importlib
import sys

_import_structure = {
    "configuration_deimv2": ["Deimv2Config"],
    "image_processing_deimv2": ["Deimv2ImageProcessor"],
    "modeling_deimv2": ["Deimv2Model", "Deimv2ForObjectDetection"],
}

# Provide a lazy module so imports are fast and consistent with HF style.
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
