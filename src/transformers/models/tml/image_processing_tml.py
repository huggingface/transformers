# Copyright 2026 the HuggingFace Team. All rights reserved.
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


import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import SizeDict, ImageInput,validate_preprocess_arguments, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


# Slightly different from `image_transforms.divide_to_patches`
def divide_to_patches(image: "torch.Tensor", patch_size: int) -> list["torch.Tensor"]:
    height, width = image.shape[-2], image.shape[-1]
    num_rows = (height + patch_size - 1) // patch_size
    num_cols = width // patch_size + 1

    patches = []
    for i in range(num_rows):
        for j in range(num_cols):
            y_base = i * patch_size
            x_base = j * patch_size
            patch = image[..., y_base : y_base + patch_size, x_base : x_base + patch_size]
            patches.append(patch)
    return patches


@auto_docstring(custom_intro="Constructs a Tml image processor.")
class TmlImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = None
    default_to_square = False
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    size = {"height": 40, "width": 40}
    valid_kwargs = ImagesKwargs

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def _validate_preprocess_kwargs(
        self,
        do_rescale: bool | None = None,
        rescale_factor: float | None = None,
        do_normalize: bool | None = None,
        image_mean: float | tuple[float] | None = None,
        image_std: float | tuple[float] | None = None,
        do_resize: bool | None = None,
        size: SizeDict | None = None,
        do_center_crop: bool | None = None,
        crop_size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ):
        """
        Validate the kwargs for the preprocess method.
        """
        if do_resize is False:
            raise ValueError("`do_resize` cannot be set to `False` for this model!")

        if size.height != size.width:
            raise ValueError(f"Can resize only to square size but got {size}")
    
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        per_image_patches: list[torch.Tensor] = []
        num_patches: list[int] = []
        for image in images:
            image_patches = divide_to_patches(image, size.height)
            image_patches = [im.float() for im in image_patches] # so we can pad with -1.0
            image_patches = self.pad(image_patches, pad_size=SizeDict(height=size.height, width=size.width), fill_value=-1.0)
            image_patches = torch.stack(image_patches, dim=0)
            image_patches = self.rescale_and_normalize(
                image_patches, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            image_patches = image_patches[..., None].repeat(1, 1, 1, 1, 2)
            per_image_patches.append(image_patches)
            num_patches.append(image_patches.shape[0])

        data = {
            "pixel_values": torch.cat(per_image_patches, dim=0),
            "num_patches": torch.tensor(num_patches),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["TmlImageProcessor"]
