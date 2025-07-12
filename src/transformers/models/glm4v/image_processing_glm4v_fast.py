# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""Fast Image processor class for GLM-4.1V."""

from typing import Optional, Union

from ...image_processing_utils import (
    BatchFeature,
)
from ...image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    valid_images,
)
from ...processing_utils import Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    logging,
)
from ...video_utils import VideoInput
from .image_processing_glm4v import smart_resize


if is_torch_available():
    import torch


if is_torchvision_available():
    from ...image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


class Glm4vFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]


@auto_docstring
class Glm4vImageProcessorFast(BaseImageProcessorFast):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 112 * 112, "longest_edge": 28 * 28 * 15000}
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Glm4vFastImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Glm4vFastImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = self.size

        super().__init__(size=size, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        do_convert_rgb: bool,
        input_data_format: Optional[Union[str, ChannelDimension]],
        device: Optional[Union[str, torch.device]],
        disable_grouping: Optional[bool],
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`List[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            device (`torch.device`, *optional*):
                The device to process the images on. If unset, the device is inferred from the input images.
        """
        images = self._prepare_input_images(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )

        height, width = get_image_size(images[0], channel_dim=ChannelDimension.FIRST)
        resized_height, resized_width = height, width

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=temporal_patch_size,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                )
                stacked_images = F.resize(
                    stacked_images, size=(resized_height, resized_width), interpolation=interpolation
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            # Fused rescale and normalize
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        patches = torch.stack(processed_images, dim=0)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = patches[-1].unsqueeze(0).repeat(temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.view(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
        disable_grouping: Optional[bool] = False,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        """

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        # Make hashable for cache
        size = SizeDict(**size) if size is not None else None
        image_mean = tuple(image_mean) if image_mean is not None else None
        image_std = tuple(image_std) if image_std is not None else None

        self._validate_preprocess_kwargs(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )
        interpolation = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        if images is not None:
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        data = {}
        if images is not None:
            pixel_values, vision_grid_thws = [], []
            for image in images:
                patches, image_grid_thw = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    interpolation=interpolation,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    device=device,
                    disable_grouping=disable_grouping,
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
            pixel_values = torch.stack(pixel_values)
            vision_grid_thws = torch.tensor(vision_grid_thws)
            data.update({"pixel_values": pixel_values, "image_grid_thw": vision_grid_thws})

        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        patch_size = images_kwargs.get("patch_size", None) or self.patch_size
        merge_size = images_kwargs.get("merge_size", None) or self.merge_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            num_frames=self.temporal_patch_size,
            height=height,
            width=width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Glm4vImageProcessorFast"]
