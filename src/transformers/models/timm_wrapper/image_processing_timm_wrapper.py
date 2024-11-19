# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import to_pil_image
from ...image_utils import ImageInput, make_list_of_images
from ...utils import TensorType, logging, requires_backends
from ...utils.import_utils import is_timm_available, is_torch_available


if is_timm_available():
    import timm

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class TimmWrapperImageProcessor(BaseImageProcessor):
    """
    Wrapper class for timm models to be used within transformers.
    """

    main_input_name = "pixel_values"

    def __init__(self, **kwargs) -> None:
        requires_backends(self, "timm")
        super().__init__(**kwargs)

        pretrained_cfg = kwargs.pop("pretrained_cfg", None)
        self.data_config = timm.data.resolve_data_config(pretrained_cfg, model=None, verbose=False)
        self.val_transforms = timm.data.create_transform(**self.data_config, is_training=False)

        # useful for training, see examples/pytorch/image-classification/run_image_classification.py
        self.train_transforms = timm.data.create_transform(**self.data_config, is_training=True)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        """
        output = super().to_dict()
        output.pop("train_transforms", None)
        output.pop("val_transforms", None)
        return output

    @classmethod
    def get_image_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get the image processor dict for the model.
        """
        image_processor_filename = kwargs.pop("image_processor_filename", "config.json")
        return super().get_image_processor_dict(
            pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
        )

    def preprocess(
        self,
        images: ImageInput,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
    ) -> BatchFeature:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                    Image to preprocess. Expects a single or batch of images
            return_tensors (`str` or `TensorType`, *optional*):
                    The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
        """
        if return_tensors != "pt":
            raise ValueError(f"return_tensors for TimmWrapperImageProcessor must be 'pt', but got {return_tensors}")

        # If the input a torch tensor, then no conversion needed
        # Otherwise, we need to pass in a list of PIL images
        if not isinstance(images, torch.Tensor):
            images = self.val_transforms(images)
            # Add batch dimension if a single image
            images = images.unsqueeze(0) if images.ndim == 3 else images
        else:
            images = make_list_of_images(images)
            images = [to_pil_image(image) for image in images]
            images = torch.stack([self.val_transforms(image) for image in images])

        return BatchFeature({"pixel_values": images}, tensor_type=return_tensors)

    def save_pretrained(self, *args, **kwargs):
        # disable it to make checkpoint the same as in `timm` library.
        logger.warning_once(
            "The `save_pretrained` method is disabled for TimmWrapperImageProcessor. "
            "Image processor configuration is saved directly in `config.json` while "
            "`save_pretrained` is called for model saving."
        )
