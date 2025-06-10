from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


_import_structure = {
    "configuration_fireflies": ["FirefliesConfig"],
    "modeling_fireflies": ["FirefliesModel"],
}

if TYPE_CHECKING:
    from .configuration_fireflies import FirefliesConfig
    from .modeling_fireflies import FirefliesModel
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
