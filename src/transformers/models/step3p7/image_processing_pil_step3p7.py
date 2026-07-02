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

from ...image_processing_backends import PilBackend
from ...image_processing_utils import get_size_dict
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring
from ...utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from .image_processing_utils_step3p7 import Step3ImagePatcherMixin


# Adapted from transformers.models.step3p7.image_processing_step3p7.Step3VisionProcessorKwargs
class Step3VisionProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, `dict[str, int]` or [`~image_utils.SizeDict`], *optional*):
        Output size for cropped image patches.
    is_patch (`bool`, *optional*):
        Whether to resize images to `patch_size` instead of `size`.
    """

    patch_size: int | dict[str, int] | SizeDict | None
    is_patch: bool | None


@auto_docstring
class Step3VisionProcessorPil(Step3ImagePatcherMixin, PilBackend):
    valid_kwargs = Step3VisionProcessorKwargs
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 728, "width": 728}
    patch_size = {"height": 504, "width": 504}
    resample = PILImageResampling.BILINEAR
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    def _standardize_kwargs(
        self, patch_size: int | dict[str, int] | SizeDict | None = None, **kwargs: Unpack[Step3VisionProcessorKwargs]
    ) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(size=patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs

    def _preprocess(
        self,
        images: list,
        is_patch: bool | None = False,
        size: SizeDict | None = None,
        patch_size: SizeDict | None = None,
        **kwargs,
    ):
        if is_patch:
            size = patch_size
        return super()._preprocess(images, size=size, **kwargs)


__all__ = ["Step3VisionProcessorPil"]
