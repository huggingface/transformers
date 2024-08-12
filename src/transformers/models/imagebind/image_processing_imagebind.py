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
"""Image processor class for ImageBind."""

import math
import warnings
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    to_channel_dimension_format,
    to_pil_image,
)
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from ...utils import TensorType, is_torch_available, is_vision_available, logging, requires_backends


logger = logging.get_logger(__name__)


if is_vision_available():
    import PIL

if is_torch_available():
    import torch


# Copy from models.video_llava.image_processing_video_llava.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], PIL.Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


# Copy from models.imagebind.feature_extraction_imagebind.uniform_chunk_sampling
def uniform_chunk_sampling(
    total_duration: float, chunk_duration: float, num_chunks: int
) -> List[Tuple[Fraction, Fraction]]:
    """
    Uniformly sample `num_chunks` chunks of duration `chunk_duration` from an audio/video of total duration `total_duration`.

    Args:
        total_duration (float): Total duration of the audio/video.
        chunk_duration (float): Duration of each chunk.
        num_chunks (int): Number of chunks to sample.

    Returns:
        List[Tuple[float, float]]: List of tuples where each tuple contains the start and end time of a chunk.
    """
    chunk_duration_fraction = Fraction(chunk_duration)
    max_possible_clip_start = Fraction(max(total_duration - chunk_duration, 0))
    uniform_clip = Fraction(max_possible_clip_start / max(num_chunks - 1, 1))

    result = []
    for clip_index in range(num_chunks):
        clip_start_sec = uniform_clip * clip_index
        clip_end_sec = clip_start_sec + chunk_duration_fraction
        result.append((clip_start_sec, clip_end_sec))

    return result


# Adapted from https://github.com/facebookresearch/pytorchvideo/blob/a0a131e/pytorchvideo/transforms/functional.py#L19
def uniform_temporal_subsample(video: VideoInput, num_samples: int) -> VideoInput:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        video (`VideoInput`):
            Video to subsample.
        num_samples (`int`):
            Number of frames to sample.
    """
    num_frames = len(video)

    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = np.linspace(0, num_frames - 1, num_samples)
    indices = np.clip(indices, 0, num_frames - 1).astype(int)

    return [video[i] for i in indices]


# Adapted from https://github.com/facebookresearch/pytorchvideo/blob/1fadaef40dd393ca09680f55582399f4679fc9b7/pytorchvideo/transforms/functional.py#L92
def video_resize(
    frames: List[np.ndarray],
    size: Tuple[int, int] = 224,
    resampling: PILImageResampling = PILImageResampling.BILINEAR,
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
) -> np.ndarray:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        image (np.ndarray): A video tensor of shape (C, T, H, W) and type numpy.float32.
        size (int): The size the shorter side is scaled to.
        resample (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format of the image. If not provided, it will be the same as the input image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format of the input image. If not provided, it will be inferred.
    Returns:
        An image-like numpy array with scaled spatial dims.
    """  # noqa
    requires_backends(video_resize, ["torch"])

    # channel-first
    frames = [
        to_channel_dimension_format(frame, ChannelDimension.FIRST, input_channel_dim=input_data_format)
        for frame in frames
    ]
    # stack, to torch and reshape to num_channels, num_frames, height, width
    video = np.stack(frames)
    video = torch.from_numpy(video).contiguous()

    data_format = input_data_format if data_format is None else data_format
    video = torch.nn.functional.interpolate(video, size=size, mode=resampling.name.lower(), align_corners=False)
    frames = list(video.numpy())
    frames = [
        to_channel_dimension_format(frame, data_format, input_channel_dim=ChannelDimension.FIRST) for frame in frames
    ]

    return frames


# Same as in image_transformers.py but taking offsets like int(math.ceil((orig_height - crop_height) / 2))
def modified_center_crop(
    image: np.ndarray,
    size: Tuple[int, int],
    data_format: Optional[Union[str, ChannelDimension]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
    return_numpy: Optional[bool] = None,
) -> np.ndarray:
    """
    Crops the `image` to the specified `size` using a center crop. Note that if the image is too small to be cropped to
    the size given, it will be padded (so the returned result will always be of size `size`).

    Args:
        image (`np.ndarray`):
            The image to crop.
        size (`Tuple[int, int]`):
            The target size for the cropped image.
        data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        input_data_format (`str` or `ChannelDimension`, *optional*):
            The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            If unset, will use the inferred format of the input image.
        return_numpy (`bool`, *optional*):
            Whether or not to return the cropped image as a numpy array. Used for backwards compatibility with the
            previous ImageFeatureExtractionMixin method.
                - Unset: will return the same type as the input image.
                - `True`: will return a numpy array.
                - `False`: will return a `PIL.Image.Image` object.
    Returns:
        `np.ndarray`: The cropped image.
    """
    requires_backends(modified_center_crop, ["vision"])

    if return_numpy is not None:
        warnings.warn("return_numpy is deprecated and will be removed in v.4.33", FutureWarning)

    return_numpy = True if return_numpy is None else return_numpy

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Input image must be of type np.ndarray, got {type(image)}")

    if not isinstance(size, Iterable) or len(size) != 2:
        raise ValueError("size must have 2 elements representing the height and width of the output image")

    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    output_data_format = data_format if data_format is not None else input_data_format

    # We perform the crop in (C, H, W) format and then convert to the output format
    image = to_channel_dimension_format(image, ChannelDimension.FIRST, input_data_format)

    orig_height, orig_width = get_image_size(image, ChannelDimension.FIRST)
    crop_height, crop_width = size
    crop_height, crop_width = int(crop_height), int(crop_width)

    # In case size is odd, (image_shape[0] + size[0]) // 2 won't give the proper result.
    top = int(math.ceil((orig_height - crop_height) / 2))
    bottom = top + crop_height
    # In case size is odd, (image_shape[1] + size[1]) // 2 won't give the proper result.
    left = int(math.ceil((orig_width - crop_width) / 2))
    right = left + crop_width

    # Check if cropped area is within image boundaries
    if top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width:
        image = image[..., top:bottom, left:right]
        image = to_channel_dimension_format(image, output_data_format, ChannelDimension.FIRST)
        return image

    # Otherwise, we may need to pad if the image is too small. Oh joy...
    new_height = max(crop_height, orig_height)
    new_width = max(crop_width, orig_width)
    new_shape = image.shape[:-2] + (new_height, new_width)
    new_image = np.zeros_like(image, shape=new_shape)

    # If the image is too small, pad it with zeros
    top_pad = math.ceil((new_height - orig_height) / 2)
    bottom_pad = top_pad + orig_height
    left_pad = math.ceil((new_width - orig_width) / 2)
    right_pad = left_pad + orig_width
    new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

    top += top_pad
    bottom += top_pad
    left += left_pad
    right += left_pad

    new_image = new_image[..., max(0, top) : min(new_height, bottom), max(0, left) : min(new_width, right)]
    new_image = to_channel_dimension_format(new_image, output_data_format, ChannelDimension.FIRST)

    if not return_numpy:
        new_image = to_pil_image(new_image)

    return new_image


class ImageBindImageProcessor(BaseImageProcessor):
    r"""
    Constructs an ImageBind image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by
            `do_resize` in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"shortest_edge": 224}`):
            Size of the image after resizing. The shortest edge of the image is resized to size["shortest_edge"], with
            the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in
            the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        do_chunk (`bool`, *optional*, defaults to `True`):
            Whether to chunk the video into multiple clips.
        chunk_duration (`float`, *optional*, defaults to 2.0):
            Duration of each chunk in seconds.
        num_chunks (`int`, *optional*, defaults to 5):
            Number of chunks to sample.
        num_frames_per_chunk (`int`, *optional*, defaults to 2):
            Number of frames to sample per chunk.
        fps (`int`, *optional*, defaults to 30):
            Frame rate of the video. It's assumed that all videos have the same frame rate.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        do_chunk: bool = True,
        chunk_duration: float = 2.0,
        num_chunks: int = 5,
        num_frames_per_chunk: int = 2,
        fps: int = 30,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else {"shortest_edge": 224}
        size = get_size_dict(size, default_to_square=False)
        crop_size = crop_size if crop_size is not None else {"height": 224, "width": 224}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.do_chunk = do_chunk
        self.chunk_duration = chunk_duration
        self.num_chunks = num_chunks
        self.num_frames_per_chunk = num_frames_per_chunk
        self.fps = fps
        self._valid_processor_keys = [
            "images",
            "do_resize",
            "size",
            "resample",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_convert_rgb",
            "do_chunk",
            "chunk_duration",
            "num_chunks",
            "fps",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

        # for backwards compatibility of KOSMOS-2
        if "use_square_size" in kwargs and kwargs["use_square_size"]:
            self.size = {"height": size["shortest_edge"], "width": size["shortest_edge"]}
            # Let's remove `use_square_size` (as it is removed from #27690), so the future Kosmos-2 image processors
            # won't have this attr. being saved. (otherwise, it will enter this if branch while there is no more
            # `shortest_edge` key.
            delattr(self, "use_square_size")

    def video_resize(
        self,
        frames: List[np.ndarray],
        size: Dict[str, int],
        resampling: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> List[np.ndarray]:
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            frames[0],
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )

        return video_resize(
            frames=frames,
            size=output_size,
            resampling=resampling,
            data_format=data_format,
            input_data_format=input_data_format,
        )

    # Copied from models.clip.image_processing_clip.CLIPImageProcessor.resize
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
        """
        default_to_square = True
        if "shortest_edge" in size:
            size = size["shortest_edge"]
            default_to_square = False
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
        else:
            raise ValueError("Size must contain either 'shortest_edge' or 'height' and 'width'.")

        output_size = get_resize_output_image_size(
            image,
            size=size,
            default_to_square=default_to_square,
            input_data_format=input_data_format,
        )
        return resize(
            image,
            size=output_size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def chunk(
        self, video: VideoInput, fps: int, chunk_duration: float, num_chunks: int, num_frames_per_chunk: int
    ) -> List[VideoInput]:
        """
        Uniformly sample `num_chunks` chunks of duration `chunk_duration` from a video.

        Args:
            video (`VideoInput`):
                Video to chunk.
            fps (`int`):
                Frame rate of the video
            chunk_duration (`float`):
                Duration of each chunk.
            num_chunks (`int`):
                Number of chunks to sample.
            num_frames_per_chunk (`int`):
                Number of frames to sample per chunk.
        """
        video_duration = len(video) / fps
        if video_duration < chunk_duration:
            logger.warning_once(
                "Chunk duration is greater than audio duration. Chunks will be repeated, consider adjusting either `chunk_duration` or `num_chunks`"
                "to avoid unnecessary memory/compute usage."
            )

        all_clips_timepoints = uniform_chunk_sampling(video_duration, chunk_duration, num_chunks)

        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            video_clip = video[math.ceil(clip_timepoints[0] * fps) : math.ceil(clip_timepoints[1] * fps)]
            video_clip = uniform_temporal_subsample(video_clip, num_samples=num_frames_per_chunk)
            all_clips.append(video_clip)

        return all_clips

    def center_crop(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Center crop an image to `(size["height"], size["width"])`. If the input size is smaller than `crop_size` along
        any edge, the image is padded with 0's and then center cropped.

        Args:
            image (`np.ndarray`):
                Image to center crop.
            size (`Dict[str, int]`):
                Size of the output image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """
        size = get_size_dict(size)
        if "height" not in size or "width" not in size:
            raise ValueError(f"The size dictionary must have keys 'height' and 'width'. Got {size.keys()}")
        return modified_center_crop(
            image,
            size=(size["height"], size["width"]),
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )

    def _preprocess_image(
        self,
        images: ImageInput,
        is_video: bool = False,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
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

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        if do_resize:
            if is_video:
                images = self.video_resize(
                    frames=images, size=size, resampling=resample, input_data_format=input_data_format
                )
            else:
                images = [
                    self.resize(image=image, size=size, resample=resample, input_data_format=input_data_format)
                    for image in images
                ]

        if do_center_crop:
            images = [
                self.center_crop(image=image, size=crop_size, input_data_format=input_data_format) for image in images
            ]

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        return images

    def preprocess(
        self,
        images: Optional[ImageInput] = None,
        videos: Optional[VideoInput] = None,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        do_chunk: bool = None,
        chunk_duration: float = None,
        num_chunks: int = None,
        num_frames_per_chunk: int = None,
        fps: int = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`, *optional*):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`. Either `images` or
                `videos` must be provided.
            videos (`VideoInput`, *optional*):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`. Either `images` or
                `videos` must be provided.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            do_chunk (`bool`, *optional*, defaults to `self.do_chunk`):
                Whether to chunk the video into multiple clips.
            chunk_duration (`float`, *optional*, defaults to `self.chunk_duration`):
                Duration of each chunk in seconds.
            num_chunks (`int`, *optional*, defaults to `self.num_chunks`):
                Number of chunks to sample.
            num_frames_per_chunk (`int`, *optional*, defaults to `self.num_frames_per_chunk`):
                Number of frames to sample per chunk.
            fps (`int`, *optional*, defaults to `self.fps`):
                Frame rate of the video. It's assumed that all videos have the same frame rate.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        if images is None and videos is None:
            raise ValueError("Either `images` or `videos` must be provided.")

        if images is not None and videos is not None:
            raise ValueError("Only one of `images` or `videos` can be provided.")

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb
        do_chunk = do_chunk if do_chunk is not None else self.do_chunk
        chunk_duration = chunk_duration if chunk_duration is not None else self.chunk_duration
        num_chunks = num_chunks if num_chunks is not None else self.num_chunks
        num_frames_per_chunk = num_frames_per_chunk if num_frames_per_chunk is not None else self.num_frames_per_chunk
        fps = fps if fps is not None else self.fps

        if images is not None:
            images = make_list_of_images(images)
        if videos is not None:
            videos = make_batched_videos(videos)

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        if (videos is not None and not valid_images(videos)) or (images is not None and not valid_images(images)):
            raise ValueError(
                "Invalid input type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if images is not None:
            pixel_values = self._preprocess_image(
                images=images,
                do_resize=do_resize,
                size=size,
                resample=resample,
                do_center_crop=do_center_crop,
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                do_convert_rgb=do_convert_rgb,
                data_format=data_format,
                input_data_format=input_data_format,
            )
        else:
            pixel_values = []
            for video in videos:
                if do_chunk:
                    clips = self.chunk(
                        video=video,
                        fps=fps,
                        chunk_duration=chunk_duration,
                        num_chunks=num_chunks,
                        num_frames_per_chunk=num_frames_per_chunk,
                    )

                    _pixel_values = [
                        self._preprocess_image(
                            images=clip,
                            is_video=True,
                            do_resize=do_resize,
                            size=size,
                            resample=PILImageResampling.BILINEAR,
                            do_center_crop=do_center_crop,
                            crop_size=crop_size,
                            do_rescale=do_rescale,
                            rescale_factor=rescale_factor,
                            do_normalize=do_normalize,
                            image_mean=image_mean,
                            image_std=image_std,
                            do_convert_rgb=do_convert_rgb,
                            data_format=data_format,
                            input_data_format=input_data_format,
                        )
                        for clip in clips
                    ]
                else:
                    _pixel_values = [
                        self._preprocess_image(
                            images=video,
                            do_resize=do_resize,
                            size=size,
                            resample=resample,
                            do_center_crop=do_center_crop,
                            crop_size=crop_size,
                            do_rescale=do_rescale,
                            rescale_factor=rescale_factor,
                            do_normalize=do_normalize,
                            image_mean=image_mean,
                            image_std=image_std,
                            do_convert_rgb=do_convert_rgb,
                            data_format=data_format,
                            input_data_format=input_data_format,
                        )
                    ]

                # Avoid List[List[List[np.ndarray]]] for performance reasons
                _pixel_values = np.stack(_pixel_values)
                # Make it shape (num_chunks, num_channels, num_frames_per_chunk, height, width)
                _pixel_values = np.swapaxes(_pixel_values, 1, 2)
                pixel_values.append(_pixel_values)

        return BatchFeature(data={"pixel_values": pixel_values}, tensor_type=return_tensors)
