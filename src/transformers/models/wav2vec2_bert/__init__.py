# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_wav2vec2_bert import Wav2Vec2BertConfig
    from .modeling_wav2vec2_bert import (
        Wav2Vec2BertForAudioFrameClassification,
        Wav2Vec2BertForCTC,
        Wav2Vec2BertForSequenceClassification,
        Wav2Vec2BertForXVector,
        Wav2Vec2BertModel,
        Wav2Vec2BertPreTrainedModel,
    )
    from .processing_wav2vec2_bert import Wav2Vec2BertProcessor
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
