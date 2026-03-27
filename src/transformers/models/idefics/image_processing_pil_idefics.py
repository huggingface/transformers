# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Image processor class for Idefics."""

from ...image_processing_backends import PilBackend
from ...image_utils import ImageInput, PILImageResampling, make_flat_list_of_images
from ...processing_utils import Unpack
from ...utils import auto_docstring, is_torch_available
from ...utils.import_utils import requires
from .image_processing_idefics import IDEFICS_STANDARD_MEAN, IDEFICS_STANDARD_STD, IdeficsImageProcessorKwargs


@requires(backends=("vision", "torch", "torchvision"))
@auto_docstring
class IdeficsImageProcessorPil(PilBackend):
    valid_kwargs = IdeficsImageProcessorKwargs
    resample = PILImageResampling.BICUBIC
    image_mean = IDEFICS_STANDARD_MEAN
    image_std = IDEFICS_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    image_num_channels = 3

    def __init__(self, **kwargs: Unpack[IdeficsImageProcessorKwargs]):
        image_size = kwargs.pop("image_size", None)
        if image_size is not None:
            kwargs["size"] = {"height": image_size, "width": image_size}
        super().__init__(**kwargs)
        self.image_size = self.size.height

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[IdeficsImageProcessorKwargs],
    ):
        r"""
        transform (`Callable`, *optional*, defaults to `None`):
            A custom transform function that accepts a single image can be passed for training. If `None`,
            inference-mode transforms are applied.
        """
        transform = kwargs.pop("transform", None)
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = self.fetch_images(images)
            images = make_flat_list_of_images(images)
            images = [transform(x) for x in images]
            return torch.stack(images)
        return super().preprocess(images, **kwargs).pixel_values


__all__ = ["IdeficsImageProcessorPil"]
