# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

from huggingface_hub.dataclasses import strict

from ...utils import (
    auto_docstring,
    logging,
)
from ..pp_ocrv6_small_rec.configuration_pp_ocrv6_small_rec import PPOCRV6SmallRecConfig
from ..pp_ocrv6_small_rec.modeling_pp_ocrv6_small_rec import PPOCRV6SmallRecForTextRecognition


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-OCRv6_base_rec_safetensors")
@strict
class PPOCRV6BaseRecConfig(PPOCRV6SmallRecConfig):
    hidden_size: int = 192
    mlp_ratio: float = 4.0


@auto_docstring(custom_intro="PPOCR6BaseRec model for text recognition tasks.")
class PPOCRV6BaseRecForTextRecognition(PPOCRV6SmallRecForTextRecognition):
    pass


__all__ = [
    "PPOCRV6BaseRecForTextRecognition",
    "PPOCRV6BaseRecConfig",
    "PPOCRV6BaseRecModel",  # noqa: F822
    "PPOCRV6BaseRecEncoderWithSVTR",  # noqa: F822
    "PPOCRV6BaseRecPreTrainedModel",  # noqa: F822
]
