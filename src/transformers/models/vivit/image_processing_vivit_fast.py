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
"""Fast Image processor class for Vivit."""

from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast, DefaultFastImageProcessorKwargs
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ChannelDimension, PILImageResampling
from ...processing_utils import Unpack
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import numpy as np
    import torch


def make_batched(videos) -> list[list]:
    """Make videos batched for processing."""
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)):
        return videos
    elif isinstance(videos, (list, tuple)):
        return [videos]
    else:
        return [[videos]]


class VivitFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    """
    offset (`bool`, *optional*):
        Whether to scale the image in both negative and positive directions. If `True`, the image has its values
        rescaled by `rescale_factor` and then offset by 1. If `rescale_factor` is 1/127.5, the image is rescaled
        between [-1, 1]. If `False`, and `rescale_factor` is 1/255, the image is rescaled between [0, 1].
    """

    offset: Optional[bool]


@auto_docstring
class VivitImageProcessorFast(BaseImageProcessorFast):
    """
    Constructs a Vivit fast image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict[str, int]` *optional*, defaults to `{"shortest_edge": 256}`):
            Size of the output image after resizing. The shortest edge of the image will be resized to
            `size["shortest_edge"]` while maintaining the aspect ratio of the original image. Can be overridden by
            `size` in the `preprocess` method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by the `do_center_crop`
            parameter in the `preprocess` method.
        crop_size (`dict[str, int]`, *optional*, defaults to `{"height": 224, "width": 224}`):
            Size of the image after applying the center crop. Can be overridden by the `crop_size` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/127.5`):
            Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter
            in the `preprocess` method.
        offset (`bool`, *optional*, defaults to `True`):
            Whether to scale the image in both negative and positive directions. Can be overridden by the `offset` in
            the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `list[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
    """

    model_input_names = ["pixel_values"]
    valid_kwargs = VivitFastImageProcessorKwargs

    # Default values from the slow image processor
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 127.5
    offset = True
    do_normalize = True
    do_convert_rgb = None

    def __init__(self, **kwargs: Unpack[VivitFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    def rescale(self, image, scale: float, offset: bool = True, **kwargs) -> "torch.Tensor":
        """
        Rescale an image by a scale factor with optional offset.

        Args:
            image:
                Image to rescale. Can be numpy array or torch tensor.
            scale (`float`):
                The scaling factor to rescale pixel values by.
            offset (`bool`, *optional*, defaults to `True`):
                Whether to scale the image in both negative and positive directions.

        Returns:
            The rescaled image in the same format as the input.
        """
        # For numpy arrays, do the computation directly to avoid precision issues
        if isinstance(image, np.ndarray):
            # Use the most numerically stable approach
            if offset:
                # Compute (image * scale - 1) in one operation to minimize precision loss
                rescaled_image = np.multiply(image.astype(np.float32), scale, dtype=np.float32)
                rescaled_image = np.subtract(rescaled_image, 1.0, dtype=np.float32)
            else:
                rescaled_image = np.multiply(image.astype(np.float32), scale, dtype=np.float32)
            return rescaled_image

        # Convert to torch tensor if needed
        if not isinstance(image, torch.Tensor):
            if hasattr(image, "numpy"):
                image = torch.from_numpy(image.numpy())
            else:
                image = torch.from_numpy(image)

        # Ensure correct format
        if image.dim() == 3 and image.shape[0] == 3:  # (C, H, W)
            pass  # Already correct
        elif image.dim() == 3 and image.shape[-1] == 3:  # (H, W, C)
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        elif image.dim() == 2:  # (H, W)
            image = image.unsqueeze(0)  # (H, W) -> (1, H, W)

        # Convert to float32 for consistent precision
        image = image.to(torch.float32)

        # Use the most numerically stable approach
        if offset:
            # Compute (image * scale - 1) in one operation to minimize precision loss
            rescaled_image = torch.sub(torch.mul(image, scale), 1.0)
        else:
            rescaled_image = torch.mul(image, scale)

        return rescaled_image

    def _prepare_image_like_inputs(
        self,
        images,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, "ChannelDimension"]] = None,
        device: Optional["torch.device"] = None,
        expected_ndims: int = 3,
    ):
        """
        Override to handle video structure for ViViT.
        """
        # Make videos batched
        videos = make_batched(images)

        # Process each video (list of frames)
        processed_videos = []
        for video in videos:
            # Process each frame in the video
            processed_frames = []
            for frame in video:
                # Convert frame to tensor and process
                if hasattr(frame, "convert") and do_convert_rgb:
                    frame = frame.convert("RGB")

                # Convert to tensor
                if not isinstance(frame, torch.Tensor):
                    if hasattr(frame, "numpy"):
                        frame = torch.from_numpy(frame.numpy())
                    elif hasattr(frame, "convert"):  # PIL Image
                        frame = torch.from_numpy(np.array(frame))
                    else:
                        frame = torch.from_numpy(frame)

                # Ensure correct format
                if frame.dim() == 3 and frame.shape[0] == 3:  # (C, H, W)
                    pass  # Already correct
                elif frame.dim() == 3 and frame.shape[-1] == 3:  # (H, W, C)
                    frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                elif frame.dim() == 2:  # (H, W)
                    frame = frame.unsqueeze(0)  # (H, W) -> (1, H, W)

                if device is not None:
                    frame = frame.to(device)

                processed_frames.append(frame)

            processed_videos.append(processed_frames)

        return processed_videos

    def _preprocess(
        self,
        images,
        do_resize: bool,
        size,
        interpolation,
        do_center_crop: bool,
        crop_size,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean,
        image_std,
        disable_grouping: Optional[bool],
        return_tensors,
        offset: bool = True,
        **kwargs,
    ):
        """
        Custom preprocessing method that handles the offset parameter for rescaling.
        """
        # Process each video separately
        processed_videos = []
        for video in images:  # images is a list of videos, each video is a list of frames
            # Process each frame in the video
            processed_frames = []
            for frame in video:
                # Apply transformations to individual frame
                if do_resize:
                    frame = self.resize(image=frame, size=size, interpolation=interpolation)

                if do_center_crop:
                    frame = self.center_crop(frame, crop_size)

                # Handle rescaling with offset parameter
                if do_rescale:
                    frame = self.rescale(frame, rescale_factor, offset=offset)

                # Handle normalization
                if do_normalize:
                    frame = self.normalize(frame, image_mean, image_std)

                processed_frames.append(frame)

            # Stack frames for this video
            if return_tensors:
                video_tensor = torch.stack(processed_frames, dim=0)
                processed_videos.append(video_tensor)
            else:
                processed_videos.append(processed_frames)

        # Stack videos if returning tensors
        if return_tensors:
            processed_videos = torch.stack(processed_videos, dim=0)

        return BatchFeature(data={"pixel_values": processed_videos}, tensor_type=return_tensors)

    @auto_docstring
    def preprocess(self, images, **kwargs: Unpack[VivitFastImageProcessorKwargs]):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Video frames to preprocess. Expects a single or batch of video frames with pixel values ranging from 0
                to 255. If passing in frames with pixel values between 0 and 1, set `do_rescale=False`.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after applying resize.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to centre crop the image.
            crop_size (`dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the image after applying the centre crop.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between `[-1 - 1]` if `offset` is `True`, `[0, 1]` otherwise.
            offset (`bool`, *optional*, defaults to `self.offset`):
                Whether to scale the image in both negative and positive directions.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                    - Unset: Use the inferred channel dimension format of the input image.
        """
        return super().preprocess(images, **kwargs)


__all__ = ["VivitImageProcessorFast"]
