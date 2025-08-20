# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for DETR."""

import warnings

from ...image_transforms import rgb_to_id as _rgb_to_id
from ...utils import logging
from ...utils.import_utils import requires
from .image_processing_detr import DetrImageProcessor


logger = logging.get_logger(__name__)


def rgb_to_id(x):
    warnings.warn(
        "rgb_to_id has moved and will not be importable from this module from v5. "
        "Please import from transformers.image_transforms instead.",
        FutureWarning,
    )
    return _rgb_to_id(x)


@requires(backends=("vision",))
class DetrFeatureExtractor(DetrImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class DetrFeatureExtractor is deprecated and will be removed in version 5 of Transformers."
            " Please use DetrImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


__all__ = ["DetrFeatureExtractor"]
