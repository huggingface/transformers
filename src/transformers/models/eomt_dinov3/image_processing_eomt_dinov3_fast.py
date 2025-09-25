# coding=utf-8
# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""EoMT-DINOv3 fast image processor."""

from ..eomt.image_processing_eomt_fast import EomtImageProcessorFast


class EomtDinov3ImageProcessorFast(EomtImageProcessorFast):
    """Alias for :class:`~transformers.EomtImageProcessorFast`."""


__all__ = ["EomtDinov3ImageProcessorFast"]
