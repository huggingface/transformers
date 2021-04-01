# coding=utf-8
# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for DeiT."""


from ...utils import logging
from ..vit.feature_extraction_vit import ViTFeatureExtractor


logger = logging.get_logger(__name__)


class DeiTFeatureExtractor(ViTFeatureExtractor):
    r"""
    Constructs a DeiT feature extractor.

    :class:`~transformers.DeiTFeatureExtractor is identical to :class:`~transformers.ViTFeatureExtractor` and can be 
    used to prepare for the model one or several image(s).

    Refer to superclass :class:`~transformers.ViTFeatureExtractor` for usage examples and documentation concerning
    parameters.
    """