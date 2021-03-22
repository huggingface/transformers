# coding=utf-8
# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for ViT."""

from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToPILImage, ToTensor

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class ViTFeatureExtractor(FeatureExtractionMixin):
    r"""
    Constructs a ViT feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        image_mean (:obj:`int`, defaults to [0.5, 0.5, 0.5]):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`int`, defaults to [0.5, 0.5, 0.5]):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int`, `optional`, defaults to :obj:`224`):
            Resize the input to the given size. Only has an effect if :obj:`do_resize` is set to :obj:`True`.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_normalize=True,
        do_resize=True,
        size=224,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_mean = image_mean
        self.image_std = image_std
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.size = size

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor, List[Image.Image], List[np.ndarray], List[torch.Tensor]],
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or a PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:
            :obj:`torch.Tensor` of shape :obj:`(batch_size, num_channels, height, width)`: A PyTorch tensor containing
            the preprocessed images.
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray, torch.Tensor)):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray, torch.Tensor)):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple)) and (isinstance(images[0], (Image.Image, np.ndarray, torch.Tensor)))
        )

        # step 1: make images a list of PIL images no matter what
        if is_batched:
            if isinstance(images[0], np.ndarray):
                # PIL expects the channel dimension as last dimension
                images = [Image.fromarray(np.moveaxis(image, 0, -1)) for image in images]
            elif isinstance(images[0], torch.Tensor):
                images = [ToPILImage()(image).convert("RGB") for image in images]
        else:
            if isinstance(images, np.ndarray):
                # PIL expects the channel dimension as last dimension
                images = [Image.fromarray(np.moveaxis(images, 0, -1))]
            elif isinstance(images, torch.Tensor):
                images = [ToPILImage()(images).convert("RGB")]
            else:
                images = [images]

        # step 2: define transformations (resizing + normalization)
        transformations = []
        if self.do_resize and self.size is not None:
            transformations.append(Resize(size=(self.size, self.size)))
        if self.do_normalize:
            normalization = Compose([ToTensor(), Normalize(self.image_mean, self.image_std)])
            transformations.append(normalization)
        transforms = Compose(transformations)

        # step 3: apply transformations to images and stack
        pixel_values = [transforms(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # return as BatchFeature
        data = {"pixel_values": pixel_values}
        encoded_inputs = BatchFeature(data=data)

        return encoded_inputs
