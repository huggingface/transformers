# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_layoutlmv2": ["LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMv2Config"],
    "processing_layoutlmv2": ["LayoutLMv2Processor"],
    "tokenization_layoutlmv2": ["LayoutLMv2Tokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_layoutlmv2_fast"] = ["LayoutLMv2TokenizerFast"]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_layoutlmv2"] = ["LayoutLMv2FeatureExtractor"]
    _import_structure["image_processing_layoutlmv2"] = ["LayoutLMv2ImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlmv2"] = [
        "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv2ForQuestionAnswering",
        "LayoutLMv2ForSequenceClassification",
        "LayoutLMv2ForTokenClassification",
        "LayoutLMv2Layer",
        "LayoutLMv2Model",
        "LayoutLMv2PreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_layoutlmv2 import LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMv2Config
    from .processing_layoutlmv2 import LayoutLMv2Processor
    from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor, LayoutLMv2ImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_layoutlmv2 import (
            LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Layer,
            LayoutLMv2Model,
            LayoutLMv2PreTrainedModel,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
