# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_bigscience176b": ["BIGSCIENCE176B_PRETRAINED_CONFIG_ARCHIVE_MAP", "BigScience176BConfig", "BigScience176BOnnxConfig"],
    "tokenization_bigscience176b": ["BigScience176BTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_bigscience176b_fast"] = ["BigScience176BTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_bigscience176b"] = [
        "BIGSCIENCE176B_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BigScience176BDoubleHeadsModel",
        "BigScience176BForSequenceClassification",
        "BigScience176BForTokenClassification",
        "BigScience176BLMHeadModel",
        "BigScience176BModel",
        "BigScience176BPreTrainedModel",
        "load_tf_weights_in_bigscience176b",
    ]

if TYPE_CHECKING:
    from .configuration_bigscience176b import BIGSCIENCE176B_PRETRAINED_CONFIG_ARCHIVE_MAP, BigScience176BConfig, BigScience176BOnnxConfig
    from .tokenization_bigscience176b import BigScience176BTokenizer

    if is_tokenizers_available():
        from .tokenization_bigscience176b_fast import BigScience176BTokenizerFast

    if is_torch_available():
        from .modeling_bigscience176b import (
            BIGSCIENCE176B_PRETRAINED_MODEL_ARCHIVE_LIST,
            BigScience176BDoubleHeadsModel,
            BigScience176BForSequenceClassification,
            BigScience176BForTokenClassification,
            BigScience176BLMHeadModel,
            BigScience176BModel,
            BigScience176BPreTrainedModel,
            load_tf_weights_in_bigscience176b,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
