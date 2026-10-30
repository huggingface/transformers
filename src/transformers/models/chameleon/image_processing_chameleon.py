# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Chameleon."""

from ...image_processing_backends import TorchvisionBackend
from ...image_utils import PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class ChameleonImageProcessor(TorchvisionBackend):
    """Torchvision backend for Chameleon."""

    resample = PILImageResampling.LANCZOS
    image_mean = [1.0, 1.0, 1.0]
    image_std = [1.0, 1.0, 1.0]
    size = {"shortest_edge": 512}
    default_to_square = False
    crop_size = {"height": 512, "width": 512}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 0.0078
    do_normalize = True
    do_convert_rgb = True

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)


__all__ = ["ChameleonImageProcessor"]
