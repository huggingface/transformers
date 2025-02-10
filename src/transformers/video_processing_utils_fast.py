# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from typing import Dict, Iterable, List, Optional, TypedDict, Union

import numpy as np

from .image_processing_utils import (
    BatchFeature,
    get_size_dict,
)
from .image_transforms import (
    get_resize_output_image_size,
    get_size_with_aspect_ratio,
)
from .image_utils import (
    ChannelDimension,
    SizeDict,
    get_image_size_for_max_height_width,
    validate_fast_preprocess_arguments,
    validate_kwargs,
)
from .processing_utils import Unpack
from .utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)
from .video_processing_utils import BaseVideoProcessor
from .video_utils import (
    VideoInput,
    group_videos_by_shape,
    make_batched_videos,
    reorder_videos,
    to_channel_dimension_format,
)


if is_vision_available():
    from .image_utils import PILImageResampling

if is_torch_available():
    import torch

if is_torchvision_available():
    from .image_utils import pil_torch_interpolation_mapping

    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

logger = logging.get_logger(__name__)


class DefaultFastVideoProcessorInitKwargs(TypedDict, total=False):
    do_resize: Optional[bool]
    size: Optional[Dict[str, int]]
    default_to_square: Optional[bool]
    resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]]
    do_center_crop: Optional[bool]
    crop_size: Optional[Dict[str, int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[Union[int, float]]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, List[float]]]
    image_std: Optional[Union[float, List[float]]]
    do_convert_rgb: Optional[bool]


class DefaultFastVideoProcessorPreprocessKwargs(DefaultFastVideoProcessorInitKwargs):
    return_tensors: Optional[Union[str, TensorType]]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional["torch.device"]


BASE_VIDEO_PROCESSOR_FAST_DOCSTRING = r"""
    Args:
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `self.size`):
            Size of the output videoafter resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        default_to_square (`bool`, *optional*, defaults to `self.default_to_square`):
            Whether to default to a square videowhen resizing, if size is an int.
        resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the videoto the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to `self.crop_size`):
            Size of the output videoafter applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the videoby the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `self.rescale_factor`):
            Scale factor to use if rescaling the video. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Mean to use if normalizing the video. This is a float or list of floats the length of the number of
            channels in the video. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Standard deviation to use if normalizing the video. This is a float or list of floats the length of the
            number of channels in the video. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `self.image_std`):
            Whether to convert the videoto RGB."""

BASE_VIDEO_PROCESSOR_FAST_DOCSTRING_PREPROCESS = r"""
    Preprocess a video or batch of videos.

    Args:
        videos (`VideoInput`):
            Image to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
            passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
        do_resize (`bool`, *optional*, defaults to `self.do_resize`):
            Whether to resize the video.
        size (`Dict[str, int]`, *optional*, defaults to `self.size`):
            Describes the maximum input dimensions to the model.
        resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `self.resample`):
            Resampling filter to use if resizing the video. This can be one of the enum `PILImageResampling`. Only
            has an effect if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
            Whether to center crop the video.
        crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
            Size of the output videoafter applying `center_crop`.
        do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
            Whether to rescale the video.
        rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
            Rescale factor to rescale the videoby if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
            Whether to normalize the video.
        image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
            Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
        image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
            Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
            `True`.
        do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
            Whether to convert the videoto RGB.
        return_tensors (`str` or `TensorType`, *optional*):
            Returns stacked tensors if set to `pt, otherwise returns a list of tensors.
        data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
            The channel dimension format for the output video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: videoin (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: videoin (height, width, num_channels) format.
            - Unset: Use the channel dimension format of the input video.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input video. If unset, the channel dimension format is inferred
            from the input video. Can be one of:
            - `"channels_first"` or `ChannelDimension.FIRST`: videoin (num_channels, height, width) format.
            - `"channels_last"` or `ChannelDimension.LAST`: videoin (height, width, num_channels) format.
            - `"none"` or `ChannelDimension.NONE`: videoin (height, width) format.
        device (`torch.device`, *optional*):
            The device to process the videos on. If unset, the device is inferred from the input videos."""


@add_start_docstrings(
    "Constructs a fast base VideoProcessor.",
    BASE_VIDEO_PROCESSOR_FAST_DOCSTRING,
)
class BaseVideoProcessorFast(BaseVideoProcessor):
    resample = None
    image_mean = None
    image_std = None
    size = None
    default_to_square = True
    crop_size = None
    do_resize = None
    do_center_crop = None
    do_rescale = None
    rescale_factor = 1 / 255
    do_normalize = None
    do_convert_rgb = None
    model_input_names = ["pixel_values_videos"]
    valid_init_kwargs = DefaultFastVideoProcessorInitKwargs
    valid_preprocess_kwargs = DefaultFastVideoProcessorPreprocessKwargs

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        size = kwargs.pop("size", self.size)
        self.size = (
            get_size_dict(size=size, default_to_square=kwargs.pop("default_to_square", self.default_to_square))
            if size is not None
            else None
        )
        crop_size = kwargs.pop("crop_size", self.crop_size)
        self.crop_size = get_size_dict(crop_size, param_name="crop_size") if crop_size is not None else None
        for key in self.valid_init_kwargs.__annotations__.keys():
            if kwargs.get(key) is not None:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, getattr(self, key, None))

    def resize(
        self,
        video: "torch.Tensor",
        size: SizeDict,
        interpolation: "F.InterpolationMode" = None,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize a video to `(size["height"], size["width"])`.

        Args:
            video (`torch.Tensor`):
                Video to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output video.
            resample (`InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                `InterpolationMode` filter to use when resizing the video e.g. `InterpolationMode.BICUBIC`.

        Returns:
            `torch.Tensor`: The resized video.
        """
        interpolation = interpolation if interpolation is not None else F.InterpolationMode.BILINEAR
        if size.shortest_edge and size.longest_edge:
            # Resize the video so that the shortest edge or the longest edge is of the given size
            # while maintaining the aspect ratio of the original video.
            new_size = get_size_with_aspect_ratio(
                video.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.shortest_edge:
            new_size = get_resize_output_image_size(
                video,
                size=size.shortest_edge,
                default_to_square=False,
                input_data_format=ChannelDimension.FIRST,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(video.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                "Size must contain 'height' and 'width' keys, or 'max_height' and 'max_width', or 'shortest_edge' key. Got"
                f" {size}."
            )
        return F.resize(video, new_size, interpolation=interpolation)

    def rescale(
        self,
        video: "torch.Tensor",
        scale: float,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Rescale a video by a scale factor. video = video * scale.

        Args:
            video (`torch.Tensor`):
                Video to rescale.
            scale (`float`):
                The scaling factor to rescale pixel values by.

        Returns:
            `torch.Tensor`: The rescaled video.
        """
        return video * scale

    def normalize(
        self,
        video: "torch.Tensor",
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Normalize a video. video = (video - mean) / std.

        Args:
            video (`torch.Tensor`):
                video to normalize.
            mean (`torch.Tensor`, `float` or `Iterable[float]`):
                video mean to use for normalization.
            std (`torch.Tensor`, `float` or `Iterable[float]`):
                video standard deviation to use for normalization.

        Returns:
            `torch.Tensor`: The normalized video.
        """
        return F.normalize(video, mean, std)

    def rescale_and_normalize(
        self,
        videos: "torch.Tensor",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Union[float, List[float]],
        image_std: Union[float, List[float]],
    ) -> "torch.Tensor":
        """
        Rescale and normalize videos.
        """
        if do_rescale and do_normalize:
            videos = self.normalize(videos.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            videos = videos * rescale_factor
        elif do_normalize:
            videos = self.normalize(videos, image_mean, image_std)

        return videos

    def center_crop(
        self,
        video: "torch.Tensor",
        size: Dict[str, int],
        **kwargs,
    ) -> "torch.Tensor":
        """
        Center crop a video to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the video is padded with 0's and then center cropped.

        Args:
            video (`"torch.Tensor"`):
                Video to center crop.
            size (`Dict[str, int]`):
                Size of the output video.

        Returns:
            `torch.Tensor`: The center cropped video.
        """
        if size.height is None or size.width is None:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return F.center_crop(video, (size["height"], size["width"]))

    def convert_to_rgb(
        self,
        video: "torch.Tensor",
    ) -> VideoInput:
        """
        Converts a video to RGB format.

        Args:
            video (`"torch.Tensor"`):
                The video to convert.

        Returns:
            `torch.Tensor`: The converted video.
        """

        video = F.grayscale_to_rgb(video)
        if video.shape[-3] == 3 or not (video[..., 3, :, :] < 255).any():
            return video

        # There is a transparency layer, blend it with a white background.
        # Calculate the alpha proportion for blending.
        alpha = video[..., 3, :, :] / 255.0
        video = (1 - alpha[..., None, :, :]) * 255 + alpha[..., None, :, :] * video[..., :3, :, :]
        return video

    def _prepare_videos_structure(
        self,
        videos: VideoInput,
    ) -> VideoInput:
        """
        Prepare the videos structure for processing by making sure the inputs
        are in correct format and converting list of `PIL.Image` frames to
        one video array.

        Args:
            videos (`VideoInput`):
                The input videos to process.

        Returns:
            `VideoInput`: A list of video arrays in np.ndarray or `torch.tensor`
        """
        return make_batched_videos(videos)

    def _process_video(
        self,
        video: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.Tensor":
        if isinstance(video, np.ndarray):
            video = to_channel_dimension_format(video, ChannelDimension.FIRST, input_data_format)
            # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
            video = torch.from_numpy(video).contiguous()

        # Now that we have torch tensors, we can move them to the right device
        if device is not None:
            video = video.to(device)

        return video

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> List["torch.Tensor"]:
        """
        Prepare the input videos for processing.
        """
        videos = self._prepare_videos_structure(videos)
        process_image_fn = partial(
            self._process_video,
            input_data_format=input_data_format,
            device=device,
        )
        with ThreadPoolExecutor() as executor:
            processed_videos = list(executor.map(process_image_fn, videos))

        return processed_videos

    @lru_cache(maxsize=10)
    def _prepare_process_arguments(
        self,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: Optional[Union["PILImageResampling", "F.InterpolationMode"]] = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        device: Optional["torch.device"] = None,
    ) -> tuple:
        """
        Prepare the arguments for the process method.
        """
        validate_fast_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            resample=resample,
            return_tensors=return_tensors,
            data_format=data_format,
        )

        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean, device=device) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std, device=device) * (1.0 / rescale_factor)

        interpolation = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        return image_mean, image_std, interpolation

    @add_start_docstrings(BASE_VIDEO_PROCESSOR_FAST_DOCSTRING_PREPROCESS)
    def preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[DefaultFastVideoProcessorPreprocessKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(), valid_processor_keys=self.valid_preprocess_kwargs.__annotations__.keys()
        )

        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_preprocess_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input videos
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")

        videos = self._prepare_input_videos(videos=videos, input_data_format=input_data_format, device=device)

        # Pop kwargs that need further processing or won't be used in _preprocess
        default_to_square = kwargs.pop("default_to_square")
        size = kwargs.pop("size")
        crop_size = kwargs.pop("crop_size")
        image_mean = kwargs.pop("image_mean")
        image_std = kwargs.pop("image_std")
        data_format = kwargs.pop("data_format")
        resample = kwargs.pop("resample")
        do_convert_rgb = kwargs.pop("do_convert_rgb")

        # Make hashable for cache
        size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square)) if size is not None else None
        crop_size = SizeDict(**get_size_dict(crop_size, param_name="crop_size")) if crop_size is not None else None
        image_mean = tuple(image_mean) if isinstance(image_mean, list) else image_mean
        image_std = tuple(image_std) if isinstance(image_std, list) else image_std

        image_mean, image_std, interpolation = self._prepare_process_arguments(
            size=size,
            crop_size=crop_size,
            resample=resample,
            image_mean=image_mean,
            image_std=image_std,
            data_format=data_format if data_format is not None else ChannelDimension.FIRST,
            device=videos[0].device,
            do_resize=kwargs.get("do_resize"),
            do_center_crop=kwargs.get("do_center_crop"),
            do_rescale=kwargs.get("do_rescale"),
            rescale_factor=kwargs.get("rescale_factor"),
            do_normalize=kwargs.get("do_normalize"),
            return_tensors=kwargs.get("return_tensors"),
        )

        return self._preprocess(
            videos=videos,
            do_convert_rgb=do_convert_rgb,
            size=size,
            crop_size=crop_size,
            interpolation=interpolation,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

    def _preprocess(
        self,
        videos: List["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        # Group videos by size for batched resizing
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(video=stacked_videos, size=size, interpolation=interpolation)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_videos = torch.stack(processed_videos, dim=0) if return_tensors else processed_videos

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)

    def to_dict(self):
        encoder_dict = super().to_dict()
        encoder_dict.pop("_valid_processor_keys", None)
        return encoder_dict
