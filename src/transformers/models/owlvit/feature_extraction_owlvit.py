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
"""Feature extractor class for OwlViT."""

from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


def center_to_corners_format(x):
    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (left, top, right, bottom).
    """
    x_center, y_center, width, height = x.unbind(-1)
    boxes = [(x_center - 0.5 * width), (y_center - 0.5 * height), (x_center + 0.5 * width), (y_center + 0.5 * height)]
    return torch.stack(boxes, dim=-1)


class OwlViTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs an OWL-ViT feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the shorter edge of the input to a certain `size`.
        size (`int`, *optional*, defaults to 768):
            Resize the shorter edge of the input to the given size. Only has an effect if `do_resize` is set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BICUBIC`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the
            image is padded with 0's and then center cropped.
        crop_size (`int`, *optional*, defaults to 768):
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
        size=768,
        resample=Image.BICUBIC,
        crop_size=768,
        do_center_crop=True,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.size = size
        self.resample = resample
        self.crop_size = crop_size
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std if image_std is not None else [0.26862954, 0.26130258, 0.27577711]

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
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        prob = nn.functional.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W) or (H, W, C),
                where C is a number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + center cropping + normalization)
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample, default_to_square=False)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
