from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_hunyuan_v1_dense import HunYuanDenseV1Config
    from .modeling_hunyuan_v1_dense import (
        HunYuanDenseV1ForCausalLM,
        HunYuanDenseV1ForSequenceClassification,
        HunYuanDenseV1Model,
        HunYuanDenseV1PreTrainedModel,
    )
    from .tokenization_hy import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
