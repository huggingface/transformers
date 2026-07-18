# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Torchvision image processor class for Apertus 1.5."""

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import ImageInput, PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class Apertus1p5ImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `256 * 256`):
        Minimum pixel area; smaller images are upscaled to approximately this area.
    max_pixels (`int`, *optional*, defaults to `1400 * 1400`):
        Maximum pixel area; larger images are downscaled to approximately this area.
    spatial_factor (`int`, *optional*, defaults to 16):
        Both output sides are rounded to multiples of this factor (the vision tokenizer's downsampling factor);
        each `spatial_factor x spatial_factor` patch becomes one discrete code.
    """

    min_pixels: int
    max_pixels: int
    spatial_factor: int


def smart_resize(
    height: int, width: int, factor: int = 16, min_pixels: int = 256 * 256, max_pixels: int = 1400 * 1400
):
    """
    Computes the output size for an image: the pixel area is clamped to `[min_pixels, max_pixels]` preserving
    the aspect ratio, and both sides are rounded half-up to multiples of `factor`.

    This reproduces the reference Apertus 1.5 pipeline exactly (including the `int()` truncations); the only
    deviation is flooring each side at `factor` so extreme aspect ratios cannot round a side down to zero.
    """
    target_area = max(min(max_pixels, height * width), min_pixels)
    aspect_ratio = width / height
    new_height = int((target_area / aspect_ratio) ** 0.5)
    new_width = int(new_height * aspect_ratio)
    new_height = ((new_height + factor // 2) // factor) * factor
    new_width = ((new_width + factor // 2) // factor) * factor
    return max(new_height, factor), max(new_width, factor)


@auto_docstring(
    custom_intro="""
    Constructs the Apertus 1.5 image processor. Input images are expected UNSCALED (PIL images or uint8-range
    pixel values; per the standard `do_rescale` convention, float images already scaled to `[0, 1]` would be
    rescaled again). Images are converted to RGB, resized preserving the aspect ratio to multiples of
    `spatial_factor` within the `[min_pixels, max_pixels]` area budget, and normalized to `[-1, 1]`
    (`pixel / 127.5 - 1`) in float32, reproducing the reference Apertus 1.5 pipeline.
    """
)
class Apertus1p5ImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    # rescale to [0, 1] then (x - 0.5) / 0.5 is exactly pixel / 127.5 - 1, the tokenizer's expected [-1, 1] range
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    do_pad = True
    min_pixels = 256 * 256
    max_pixels = 1400 * 1400
    spatial_factor = 16
    valid_kwargs = Apertus1p5ImageProcessorKwargs
    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(self, **kwargs: Unpack[Apertus1p5ImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(self, do_resize=None, size=None, **kwargs):
        # resizing is governed by `min_pixels` / `max_pixels` / `spatial_factor`; a `size` dict is accepted
        # for API compatibility but ignored
        super()._validate_preprocess_kwargs(**kwargs)

    @auto_docstring
    def preprocess(self, images: ImageInput, **kwargs: Unpack[Apertus1p5ImageProcessorKwargs]) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool,
        min_pixels: int,
        max_pixels: int,
        spatial_factor: int,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                height, width = stacked_images.shape[-2:]
                resized_height, resized_width = smart_resize(
                    height, width, factor=spatial_factor, min_pixels=min_pixels, max_pixels=max_pixels
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        # true per-image sizes; the padded regions below carry no information and the model crops them away
        image_sizes = [list(image.shape[-2:]) for image in processed_images]
        # per-image code-grid sizes, consumed by the processor's placeholder expansion (not a model input)
        image_grids = [[height // spatial_factor, width // spatial_factor] for height, width in image_sizes]
        if do_pad:
            processed_images = self.pad(processed_images, disable_grouping=disable_grouping)
            processed_images = torch.stack(processed_images, dim=0)

        return BatchFeature(
            data={
                "pixel_values": processed_images,
                "image_sizes": torch.tensor(image_sizes, dtype=torch.long),
                "image_grids": torch.tensor(image_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns the number of discrete image codes (placeholder tokens) for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image codes per image.
        """
        images_kwargs = images_kwargs if images_kwargs is not None else {}
        spatial_factor = images_kwargs.get("spatial_factor", self.spatial_factor)
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)

        resized_height, resized_width = smart_resize(
            height, width, factor=spatial_factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return (resized_height // spatial_factor) * (resized_width // spatial_factor)


__all__ = ["Apertus1p5ImageProcessor"]
