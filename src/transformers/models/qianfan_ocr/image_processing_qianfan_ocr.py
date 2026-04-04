# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for QianfanOCR (Torchvision backend)."""

from ...processing_utils import Unpack
from ...utils import auto_docstring
from ..got_ocr2.image_processing_got_ocr2 import GotOcr2ImageProcessor, GotOcr2ImageProcessorKwargs


QianfanOCRImageProcessorKwargs = GotOcr2ImageProcessorKwargs


@auto_docstring
class QianfanOCRImageProcessor(GotOcr2ImageProcessor):
    def __init__(self, **kwargs: Unpack[QianfanOCRImageProcessorKwargs]):
        super().__init__(**kwargs)


__all__ = ["QianfanOCRImageProcessor"]
