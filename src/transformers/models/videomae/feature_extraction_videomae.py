# coding=utf-8
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
"""Feature extractor class for VideoMAE."""

from typing import Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, ImageInput, is_torch_tensor
from ...utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, TensorType, logging


logger = logging.get_logger(__name__)


class VideoMAEFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a VideoMAE feature extractor. This feature extractor can be used to prepare videos for the model.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`int`, *optional*, defaults to 224):
            Resize the shorter edge of the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the input to a certain `size`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        do_center_crop=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD

    def resize_video(self, video, size, resample="bilinear"):
        return [self.resize(frame, size, resample, default_to_square=False) for frame in video]

    def crop_video(self, video, size):
        return [self.center_crop(frame, size) for frame in video]

    def normalize_video(self, video, mean, std):
        # video can be a list of PIL images, list of NumPy arrays or list of PyTorch tensors
        # first: convert to list of NumPy arrays
        video = [self.to_numpy_array(frame) for frame in video]

        # second: stack to get (num_frames, num_channels, height, width)
        video = np.stack(video, axis=0)

        # third: normalize
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(video.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(video.dtype)

        return (video - mean[None, :, None, None]) / std[None, :, None, None]

    def __call__(
        self, videos: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several video(s).

        <Tip warning={true}>

        NumPy arrays are converted to PIL images when resizing, so the most efficient is to pass PIL images.

        </Tip>

        Args:
            videos (`List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, `List[List[PIL.Image.Image]]`, `List[List[np.ndarrray]]`,:
                `List[List[torch.Tensor]]`): The video or batch of videos to be prepared. Each video should be a list
                of frames, which can be either PIL images or NumPy arrays. In case of NumPy arrays/PyTorch tensors,
                each frame should be of shape (H, W, C), where H and W are frame height and width, and C is a number of
                channels.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, num_frames,
              height, width).
        """
        # Input type checking for clearer error
        valid_videos = False
        is_batched = False

        # Check that videos have a valid type
        if isinstance(videos, (list, tuple)):
            if isinstance(videos[0], (Image.Image, np.ndarray)) or is_torch_tensor(videos[0]):
                valid_videos = True
            elif isinstance(videos[0], (list, tuple)) and (
                isinstance(videos[0][0], (Image.Image, np.ndarray)) or is_torch_tensor(videos[0][0])
            ):
                valid_videos = True
                is_batched = True

        if not valid_videos:
            raise ValueError(
                "Videos must of type `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]` (single"
                " example), `List[List[PIL.Image.Image]]`, `List[List[np.ndarray]]`, `List[List[torch.Tensor]]` (batch"
                " of examples)."
            )

        if not is_batched:
            videos = [videos]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None:
            videos = [self.resize_video(video, size=self.size, resample=self.resample) for video in videos]
        if self.do_center_crop and self.size is not None:
            videos = [self.crop_video(video, size=self.size) for video in videos]
        if self.do_normalize:
            videos = [self.normalize_video(video, mean=self.image_mean, std=self.image_std) for video in videos]

        # return as BatchFeature
        data = {"pixel_values": videos}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
