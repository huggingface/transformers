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

from ...file_utils import _LazyModule, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_dpr": ["DPR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPRConfig"],
    "tokenization_dpr": [
        "DPRContextEncoderTokenizer",
        "DPRQuestionEncoderTokenizer",
        "DPRReaderOutput",
        "DPRReaderTokenizer",
    ],
}


if is_tokenizers_available():
    _import_structure["tokenization_dpr_fast"] = [
        "DPRContextEncoderTokenizerFast",
        "DPRQuestionEncoderTokenizerFast",
        "DPRReaderTokenizerFast",
    ]

if is_torch_available():
    _import_structure["modeling_dpr"] = [
        "DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPRContextEncoder",
        "DPRPretrainedContextEncoder",
        "DPRPreTrainedModel",
        "DPRPretrainedQuestionEncoder",
        "DPRPretrainedReader",
        "DPRQuestionEncoder",
        "DPRReader",
    ]

if is_tf_available():
    _import_structure["modeling_tf_dpr"] = [
        "TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFDPRContextEncoder",
        "TFDPRPretrainedContextEncoder",
        "TFDPRPretrainedQuestionEncoder",
        "TFDPRPretrainedReader",
        "TFDPRQuestionEncoder",
        "TFDPRReader",
    ]


if TYPE_CHECKING:
    from .configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
    from .tokenization_dpr import (
        DPRContextEncoderTokenizer,
        DPRQuestionEncoderTokenizer,
        DPRReaderOutput,
        DPRReaderTokenizer,
    )

    if is_tokenizers_available():
        from .tokenization_dpr_fast import (
            DPRContextEncoderTokenizerFast,
            DPRQuestionEncoderTokenizerFast,
            DPRReaderTokenizerFast,
        )

    if is_torch_available():
        from .modeling_dpr import (
            DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPRContextEncoder,
            DPRPretrainedContextEncoder,
            DPRPreTrainedModel,
            DPRPretrainedQuestionEncoder,
            DPRPretrainedReader,
            DPRQuestionEncoder,
            DPRReader,
        )

    if is_tf_available():
        from .modeling_tf_dpr import (
            TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFDPRContextEncoder,
            TFDPRPretrainedContextEncoder,
            TFDPRPretrainedQuestionEncoder,
            TFDPRPretrainedReader,
            TFDPRQuestionEncoder,
            TFDPRReader,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
