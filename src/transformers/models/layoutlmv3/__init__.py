# Copyright 2022 The HuggingFace Team. All rights reserved.
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
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {
    "configuration_layoutlmv3": [
        "LayoutLMv3Config",
        "LayoutLMv3OnnxConfig",
    ],
    "processing_layoutlmv3": ["LayoutLMv3Processor"],
    "tokenization_layoutlmv3": ["LayoutLMv3Tokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_layoutlmv3_fast"] = ["LayoutLMv3TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlmv3"] = [
        "LayoutLMv3ForQuestionAnswering",
        "LayoutLMv3ForSequenceClassification",
        "LayoutLMv3ForTokenClassification",
        "LayoutLMv3Model",
        "LayoutLMv3PreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_layoutlmv3"] = [
        "TFLayoutLMv3ForQuestionAnswering",
        "TFLayoutLMv3ForSequenceClassification",
        "TFLayoutLMv3ForTokenClassification",
        "TFLayoutLMv3Model",
        "TFLayoutLMv3PreTrainedModel",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_layoutlmv3"] = ["LayoutLMv3FeatureExtractor"]
    _import_structure["image_processing_layoutlmv3"] = ["LayoutLMv3ImageProcessor"]


if TYPE_CHECKING:
    from .configuration_layoutlmv3 import (
        LayoutLMv3Config,
        LayoutLMv3OnnxConfig,
    )
    from .processing_layoutlmv3 import LayoutLMv3Processor
    from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_layoutlmv3 import (
            LayoutLMv3ForQuestionAnswering,
            LayoutLMv3ForSequenceClassification,
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Model,
            LayoutLMv3PreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_layoutlmv3 import (
            TFLayoutLMv3ForQuestionAnswering,
            TFLayoutLMv3ForSequenceClassification,
            TFLayoutLMv3ForTokenClassification,
            TFLayoutLMv3Model,
            TFLayoutLMv3PreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_layoutlmv3 import LayoutLMv3FeatureExtractor
        from .image_processing_layoutlmv3 import LayoutLMv3ImageProcessor

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
