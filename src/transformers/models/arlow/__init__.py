# src/transformers/models/arlow/__init__.py

from typing import TYPE_CHECKING

from ...utils import _LazyModule


########################################################################
# 1) Build _import_structure
########################################################################
_import_structure = {
    "configuration_arlow": ["ArlowConfig"],
    "modeling_arlow": [
        "ArlowForCausalLM",
        "ArlowModel",
        "ArlowPreTrainedModel",
    ],
    "tokenization_arlow": ["ArlowTokenizer"],
}

########################################################################
# 2) TYPE_CHECKING => EXACT MATCH EAGER IMPORT
########################################################################
if TYPE_CHECKING:
    from .configuration_arlow import ArlowConfig
    from .modeling_arlow import (
        ArlowForCausalLM,
        ArlowModel,
        ArlowPreTrainedModel,
    )
    from .tokenization_arlow import ArlowTokenizer

########################################################################
# 3) ELSE => LAZY LOADING
########################################################################
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
