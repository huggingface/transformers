# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict, is_valid_image, load_image
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring


class Step3VisionProcessorKwargs(ImagesKwargs, total=False):
    r"""
    is_patch (`bool`, *optional*):
        Whether to resize images to `patch_size` instead of `size`.
    """

    is_patch: bool | None


@auto_docstring
class Step3VisionProcessor(TorchvisionBackend):
    valid_kwargs = Step3VisionProcessorKwargs
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 728, "width": 728}
    patch_size = {"height": 504, "width": 504}
    resample = PILImageResampling.BILINEAR
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def __init__(
        self,
        size: int | dict[str, int] | SizeDict | None = None,
        interpolation_mode: str | None = None,
        patch_size: int | dict[str, int] | SizeDict | None = None,
        **kwargs: Unpack[Step3VisionProcessorKwargs],
    ):
        r"""
        Args:
            size (`int`, `dict[str, int]` or [`~image_utils.SizeDict`], *optional*):
                Output size for full images.
            interpolation_mode (`str`, *optional*):
                Backward-compatible interpolation shortcut. Must be either `"bicubic"` or `"bilinear"`.
            patch_size (`int`, `dict[str, int]` or [`~image_utils.SizeDict`], *optional*):
                Output size for cropped image patches.
        """
        if size is not None:
            kwargs["size"] = size
        if patch_size is not None:
            kwargs["patch_size"] = patch_size
        if interpolation_mode is not None:
            kwargs["resample"] = self._interpolation_mode_to_resample(interpolation_mode)
        super().__init__(**kwargs)
        self.patch_size = self._standardize_size(self.patch_size)

    def fetch_images(self, image_url_or_urls: str | list[str] | list[list[str]]):
        if isinstance(image_url_or_urls, (list, tuple)):
            return [self.fetch_images(x) for x in image_url_or_urls]
        if isinstance(image_url_or_urls, str):
            return load_image(image_url_or_urls)
        if is_valid_image(image_url_or_urls):
            return image_url_or_urls
        raise TypeError(f"only a single or a list of entries is supported but got type={type(image_url_or_urls)}")

    @staticmethod
    def _interpolation_mode_to_resample(interpolation_mode: str) -> PILImageResampling:
        if interpolation_mode == "bicubic":
            return PILImageResampling.BICUBIC
        if interpolation_mode == "bilinear":
            return PILImageResampling.BILINEAR
        raise ValueError("`interpolation_mode` should be either 'bicubic' or 'bilinear'.")

    @staticmethod
    def _standardize_size(size: int | dict[str, int] | SizeDict) -> SizeDict:
        if isinstance(size, SizeDict):
            return size
        if isinstance(size, int):
            return SizeDict(height=size, width=size)
        return SizeDict(**size)

    def _preprocess(
        self,
        images: list[torch.Tensor],
        is_patch: bool | None = False,
        size: SizeDict | None = None,
        patch_size: SizeDict | None = None,
        **kwargs,
    ) -> BatchFeature:
        if size is None:
            size = self.size
        if patch_size is None:
            patch_size = self.patch_size
        size = patch_size if is_patch else size
        return super()._preprocess(images, size=size, **kwargs)


__all__ = ["Step3VisionProcessor"]
