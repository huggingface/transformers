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
"""Image processor class for OwlViT"""

import numpy as np

from typing import Optional, Union, Dict, List
from transformers.utils import is_torch_available, logging, TensorType
from transformers.image_transforms import center_to_corners_format, resize, center_crop, rescale, normalize, to_numpy_array, to_channel_dimension_format
from transformers.image_processing_utils import BaseImageProcessor, get_size_dict, BatchFeature
from transformers.image_utils import PILImageResampling, is_batched, valid_images, ChannelDimension, ImageInput

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class OwlViTImageProcessor(BaseImageProcessor):
    r"""
    Constructs an OWL-ViT feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`Dict[str, int]`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for resizing the image. Only has an effect if `do_resize` is set to `True`. If `size` is a
            sequence like (h, w), output size will be matched to this. If `size` is an int, then image will be resized
            to (size, size).
        resample (`int`, *optional*, defaults to `PIL.Image.Resampling.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.Resampling.NEAREST`,
            `PIL.Image.Resampling.BOX`, `PIL.Image.Resampling.BILINEAR`, `PIL.Image.Resampling.HAMMING`,
            `PIL.Image.Resampling.BICUBIC` or `PIL.Image.Resampling.LANCZOS`. Only has an effect if `do_resize` is set
            to `True`.
        do_center_crop (`bool`, *optional*, defaults to `False`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to {"height": 768, "width": 768}):
            The size to use for center cropping the image. Only has an effect if `do_center_crop` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the input by a certain factor.
        rescale_factor (`float`, *optional*, defaults to `1/255`):
            The factor to use for rescaling the image. Only has an effect if `do_rescale` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with `image_mean` and `image_std`. Desired output size when applying
            center-cropping. Only has an effect if `do_center_crop` is set to `True`.
        image_mean (`List[int]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=None,
        resample=PILImageResampling.BICUBIC,
        do_center_crop=False,
        crop_size=None,
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        size = size if size is not None else {"height": 768, "width": 768}
        size = get_size_dict(size, default_to_square=True)

        crop_size = crop_size if crop_size is not None else {"height": 768, "width": 768}
        crop_size = get_size_dict(crop_size, default_to_square=True)

        # Early versions of the OWL-ViT config on the hub had "rescale" as a flag. This clashes with the
        # vision feature extractor method `rescale` as it would be set as an attribute during the super().__init__
        # call. This is for backwards compatibility.
        if "rescale" in kwargs:
            rescale_val = kwargs.pop("rescale")
            kwargs["do_rescale"] = rescale_val

        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]

    def resize(self, image: np.ndarray, size: Dict[str, int], resample: PILImageResampling.BICUBIC, data_format: Union[str, ChannelDimension], **kwargs) -> np.ndarray:
        """
        Resize an image to a certain size.
        """
        size = get_size_dict(size, default_to_square=True)
        if "height" not in size or "width" not in size:
            raise ValueError("size dictionary must contain height and width keys")

        return resize(image, (size["height"], size["width"]), resample=resample, data_format=data_format, **kwargs)

    def center_crop(self, image: np.ndarray, crop_size: Dict[str, int], data_format: Union[str, ChannelDimension], **kwargs) -> np.ndarray:
        """
        Center crop an image to a certain size.
        """
        crop_size = get_size_dict(crop_size, default_to_square=True)
        if "height" not in crop_size or "width" not in crop_size:
            raise ValueError("crop_size dictionary must contain height and width keys")

        return center_crop(image, (crop_size["height"], crop_size["width"]), data_format=data_format, **kwargs)

    def rescale(self, image: np.ndarray, rescale_factor: float, **kwargs) -> np.ndarray:
        """
        Rescale an image by a certain factor.
        """
        return rescale(image, rescale_factor, **kwargs)

    def normalize(self, image: np.ndarray, mean: List[float], std: List[float], **kwargs) -> np.ndarray:
        """
        Normalize an image with a certain mean and standard deviation.
        """
        return normalize(image, mean, std, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional["PILImageResampling"] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[TensorType, str]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
    ) -> BatchFeature:
        """
        Prepares an image or batch of images for the model.

        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether or not to resize the input. If `True`, will resize the input to the size specified by `size`.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                The size to resize the input to. Only has an effect if `do_resize` is set to `True`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                The resampling filter to use when resizing the input. Only has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether or not to center crop the input. If `True`, will center crop the input to the size specified by
                `crop_size`.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                The size to center crop the input to. Only has an effect if `do_center_crop` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether or not to rescale the input. If `True`, will rescale the input by dividing it by `rescale_factor`.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                The factor to rescale the input by. Only has an effect if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether or not to normalize the input. If `True`, will normalize the input by subtracting `image_mean` and
                dividing by `image_std`.
            image_mean (`Union[float, List[float]]`, *optional*, defaults to `self.image_mean`):
                The mean to subtract from the input when normalizing. Only has an effect if `do_normalize` is set to `True`.
            image_std (`Union[float, List[float]]`, *optional*, defaults to `self.image_std`):
                The standard deviation to divide the input by when normalizing. Only has an effect if `do_normalize` is set to `True`.
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
                - Unset: defaults to the channel dimension format of the input image.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        if do_resize is not None and size is None:
            raise ValueError("Size and max_size must be specified if do_resize is True.")

        if do_center_crop is not None and crop_size is None:
            raise ValueError("Crop size must be specified if do_center_crop is True.")

        if do_rescale is not None and rescale_factor is None:
            raise ValueError("Rescale factor must be specified if do_rescale is True.")

        if do_normalize is not None and (image_mean is None or image_std is None):
            raise ValueError("Image mean and std must be specified if do_normalize is True.")

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # All transformations expect numpy arrays
        images = [to_numpy_array(image) for image in images]

        if do_resize:
            images = [self.resize(image, size=size, resample=resample) for image in images]

        if do_center_crop:
            images = [self.center_crop(image, crop_size=crop_size) for image in images]

        if do_rescale:
            images = [self.rescale(image, scale=rescale_factor) for image in images]

        if do_normalize:
            images = [self.normalize(image, mean=image_mean, std=image_std) for image in images]

        images = [to_channel_dimension_format(image, data_format) for image in images]
        encoded_inputs = BatchFeature(data={"pixel_values": images}, return_tensors=return_tensors)
        return encoded_inputs

    def post_process(self, outputs, target_sizes):
        """
        Converts the output of [`OwlViTForObjectDetection`] into the format expected by the COCO api.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        # TODO: (amy) add support for other frameworks
        logits, boxes = outputs.logits, outputs.pred_boxes

        if len(logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        probs = torch.max(logits, dim=-1)
        scores = torch.sigmoid(probs.values)
        labels = probs.indices

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(boxes)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results
