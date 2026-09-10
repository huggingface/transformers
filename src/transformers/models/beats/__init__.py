# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the MIT License.
from typing import TYPE_CHECKING
from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure
if TYPE_CHECKING:
    from .configuration_beats import *
    from .feature_extraction_beats import *
    from .modeling_beats import *
else:
    import sys
    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)