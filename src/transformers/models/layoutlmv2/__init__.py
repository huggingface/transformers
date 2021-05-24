from .configuration_layoutlmv2 import LayoutLMv2Config
from .modeling_layoutlmv2 import LayoutLMv2ForTokenClassification
from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer
from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

from typing import TYPE_CHECKING

from ...file_utils import _BaseLazyModule,is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_layoutlmv2": ["LAYOUTLMv2_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMv2Config"],
    "tokenization_layoutlmv2": ["LayoutLMv2Tokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_layoutlmv2_fast"] = ["LayoutLMv2TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_layoutlmv2"] = [
        "LAYOUTLMv2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv2ForTokenClassification",
        "LayoutLMv2Model",
    ]


if TYPE_CHECKING:
    from .configuration_layoutlmv2 import LAYOUTLMv2_PRETRAINED_CONFIG_ARCHIVE_MAP, Layoutv2LMConfig
    from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer

    if is_tokenizers_available():
        from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

    if is_torch_available():
        from .modeling_layoutlmv2 import (
            LAYOUTLMv2_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Model,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
