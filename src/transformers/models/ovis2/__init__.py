# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_ovis2 import *
    from .image_processing_ovis2 import *
    from .image_processing_ovis2_fast import *
    from .modeling_ovis2 import *
    from .processing_ovis2 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
