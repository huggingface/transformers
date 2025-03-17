# src/transformers/models/arlow/__init__.py

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    # is_torch_available,  # Commented out for now since we have no model
)

######################################################################
# _import_structure - Only references the tokenizer for now.
######################################################################
_import_structure = {
    "tokenization_arlow": ["ArlowTokenizer"],   # We have a real fast tokenizer here.
}

######################################################################
# If you later create a separate "tokenization_arlow_fast.py" or slow tokenizer,
# you can add it under this block, referencing is_tokenizers_available().
######################################################################
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # For a second file "tokenization_arlow_fast.py", you'd add it here, e.g.:
    # _import_structure["tokenization_arlow_fast"] = ["ArlowTokenizerFast"]
    pass

######################################################################
# MODEL-RELATED BLOCKS (COMMENTED OUT)
# Uncomment if/when you add config & modeling files, e.g. configuration_arlow.py or modeling_arlow.py
######################################################################
# try:
#     if not is_torch_available():
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     pass
# else:
#     _import_structure["configuration_arlow"] = ["ArlowConfig"]
#     _import_structure["modeling_arlow"] = [
#         "ArlowPreTrainedModel",
#         "ArlowModel",
#         "ArlowForCausalLM",
#     ]

######################################################################
# TYPE_CHECKING section - eagerly imports only the tokenizer for type checks
######################################################################
if TYPE_CHECKING:
    from .tokenization_arlow import ArlowTokenizer

    # If you add "tokenization_arlow_fast.py," import it here too:
    # from .tokenization_arlow_fast import ArlowTokenizerFast

    # MODEL IMPORTS FOR LATER:
    # from .configuration_arlow import ArlowConfig
    # from .modeling_arlow import (
    #     ArlowPreTrainedModel,
    #     ArlowModel,
    #     ArlowForCausalLM,
    # )

######################################################################
# Else: lazy-load your tokenizer to save import time.
######################################################################
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
