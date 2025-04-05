# coding=utf-8
# Copyright 2025 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for Chameleon."""

from typing import Optional, Union

import numpy as np

from ...image_processing_utils_fast import BASE_IMAGE_PROCESSOR_FAST_DOCSTRING, BaseImageProcessorFast
from ...image_transforms import get_resize_output_image_size, get_size_with_aspect_ratio, resize
from ...image_utils import (
    ChannelDimension, 
    ImageType,
    infer_channel_dimension_format,
    ImageInput, 
    SizeDict, 
    get_image_size_for_max_height_width,
    get_image_type,
)
from ...utils import (
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)

if is_vision_available():
    import PIL
if is_torch_available():
    import torch
if is_torchvision_available():
    from ...image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


@add_start_docstrings(
    "Constructs a fast Chameleon image processor.",
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
)
class ChameleonImageProcessorFast(BaseImageProcessorFast):
    # This generated class can be used as a starting point for the fast image processor.
    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,
    # only the default values should be set in the class.
    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.
    # In most cases, only the `_preprocess` method should be overridden.

    # For an example of a fast image processor requiring more complex augmentations, see `LlavaNextImageProcessorFast`.

    # Default values should be checked against the slow image processor
    # None values left after checking can be removed
    resample = PIL.Image.LANCZOS
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

    def _process_image(
        self,
        image: ImageInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        image_type = get_image_type(image)
        if image_type not in [ImageType.PIL, ImageType.TORCH, ImageType.NUMPY]:
            raise ValueError(f"Unsupported input image type {image_type}")

        if do_convert_rgb:
            image = self.blend_rgba(image)

        if image_type == ImageType.PIL:
            image = F.pil_to_tensor(image)
        elif image_type == ImageType.NUMPY:
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            image = torch.from_numpy(image).contiguous()

        # Infer the channel dimension format if not provided
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        if input_data_format == ChannelDimension.LAST:
            # We force the channel dimension to be first for torch tensors as this is what torchvision expects.
            image = image.permute(2, 0, 1).contiguous()

        # Now that we have torch tensors, we can move them to the right device
        if device is not None:
            image = image.to(device)

        return image

    def blend_rgba(self, image: ImageInput) -> ImageInput:
        """
        Convert image to RGB by blending the transparency layer if it's in RGBA format.
        If image is not `PIL.Image`, it si simply returned without modifications.

        Args:
            image (`ImageInput`):
                Image to convert.
        """

        if not isinstance(image, PIL.Image.Image):
            return image
        elif image.mode == "RGB":
            return image

        img_rgba = np.array(image.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (img_rgba[:, :, 3] < 255).any():
            return image.convert("RGB")

        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = img_rgba[:, :, 3] / 255.0
        img_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[:, :, np.newaxis] * img_rgba[:, :, :3]
        return PIL.Image.fromarray(img_rgb.astype("uint8"), "RGB")

    def resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the image e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized image.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        pil_torch_interpolation_mapping_inverse = {v: k for k, v in pil_torch_interpolation_mapping.items()}
        if isinstance(interpolation, F.InterpolationMode):
            interpolation = pil_torch_interpolation_mapping_inverse[interpolation]
        if size.shortest_edge and size.longest_edge:
            # Resize the image so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original image.
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                image,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        # resize the image one by one as torchvision does not support batch resizing with LANCZOS interpolation
        device = image.device
        image_stack = []
        for img in image:
            img = img.cpu().numpy()
            img = resize(
                img,
                new_size,
                resample=interpolation,
                input_data_format=ChannelDimension.FIRST,
                **kwargs,
            )
            img = torch.from_numpy(img).contiguous().to(device)
            image_stack.append(img)
        return torch.stack(image_stack, dim=0)


__all__ = ["ChameleonImageProcessorFast"]
