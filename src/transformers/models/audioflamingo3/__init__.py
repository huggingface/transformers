from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_audioflamingo3 import *
    from .modeling_audioflamingo3 import *
    from .processing_audioflamingo3 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
