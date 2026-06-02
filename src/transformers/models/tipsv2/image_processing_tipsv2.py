# Copyright 2026 Google LLC and the HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for TIPSv2."""

from ...image_processing_backends import TorchvisionBackend
from ...image_utils import PILImageResampling
from ...utils import auto_docstring


@auto_docstring(custom_intro="Constructs a TIPSv2 image processor.")
class Tipsv2ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    size = {"height": 448, "width": 448}
    do_resize = True
    do_rescale = True
    do_normalize = False
    do_convert_rgb = True


__all__ = ["Tipsv2ImageProcessor"]
