# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Protein processor class for Evolla."""

from typing import Callable, Dict, List, Optional, Union

from PIL import Image

from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import resize, to_channel_dimension_format
from ...image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from ...utils import TensorType, is_torch_available


EVOLLA_STANDARD_MEAN = [0.48145466, 0.4578275, 0.40821073]
EVOLLA_STANDARD_STD = [0.26862954, 0.26130258, 0.27577711]
EVOLLA_VALID_AA = list("ACDEFGHIKLMNPQRSTVWY#")
EVOLLA_VALID_FS = list("pynwrqhgdlvtmfsaeikc#")

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

class EvollaSequenceProcessor(BaseImageProcessor):
    r"""
    Constructs a Evolla amino acid sequence processor.
    
    Args:
        sequence_max_length (`int`, *optional*, defaults to 1024):
            The maximum length of the sequence.
    """
    def __init__(self, sequence_max_length: int = 1024, SaProt_tokenizer=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sequence_max_length = sequence_max_length
        self.SaProt_tokenizer = SaProt_tokenizer
    
    def preprocess(
        self,
        proteins: List[dict],
        sequence_max_length: Optional[int] = None,
        uppercase: Optional[bool] = None,
        check_validity_strategy: Optional[str] = "raise",
        **kwargs
    ):
        """
        Preprocess a list of sequences.

        Args:
            sequences (`List[str]`):
                A list of sequences to preprocess.
            sequence_max_length (`int`, *optional*, defaults to `self.sequence_max_length`):
                The maximum length of the sequence. If `None` - the default value of the processor will be used. If 
                minus or zero - the sequence will be padded to the maximum length.
            uppercase (`bool`, *optional*, defaults to `None`):
                Whether to convert the sequences to uppercase. If `None` - the sequences will not be converted to
                uppercase. If `True` - the sequences will be converted to uppercase. If `False` - the sequences will not
                be converted to uppercase.
            check_validity_strategy (`str`, *optional*, defaults to `"raise"`):
                The strategy to use for checking the validity of the sequences. Can be one of `"raise"`, `"warning"`, 
                `"ignore"`. If `"raise"` - an exception will be raised if the sequences are not valid, `"warning"` - a warning
                information will be printed, `"ignore"` - no action will be taken.

        Returns:
            a list of preprocessed sequences
        """
        assert isinstance(sequences, list), f"sequences should be a list of strings, but is {type(sequences)}"

        sequence_max_length = sequence_max_length if sequence_max_length is not None else self.sequence_max_length
        
        if sequence_max_length <= 0:
            sequence_max_length = max([len(seq) for seq in sequences])
        sequences = [seq[:sequence_max_length] for seq in sequences]

        if uppercase:
            sequences = [seq.upper() for seq in sequences]
        
        if check_validity_strategy == "raise":
            if any(s not in EVOLLA_VALID_AA for seq in sequences for s in seq):
                raise ValueError(f"Invalid amino acid found in the sequences: {sequences}")
        elif check_validity_strategy == "warning":
            if any(s not in EVOLLA_VALID_AA for seq in sequences for s in seq):
                print(f"Invalid amino acid found in the sequences: {sequences}", flush=True)
        elif check_validity_strategy == "ignore":
            pass
        else:
            raise ValueError(f"Invalid check_validity_strategy: {check_validity_strategy}")
        
        return sequences

class EvollaProteinProcessor(BaseImageProcessor):
    r"""
    Constructs a Evolla image processor.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            Resize to image size
        image_mean (`float` or `List[float]`, *optional*, defaults to `EVOLLA_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `EVOLLA_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        image_num_channels (`int`, *optional*, defaults to 3):
            Number of image channels.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 224,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        image_num_channels: Optional[int] = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.image_size = image_size
        self.image_num_channels = image_num_channels
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: ImageInput,
        image_num_channels: Optional[int] = 3,
        image_size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        transform: Callable = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        **kwargs,
    ) -> TensorType:
        """
        Preprocess a batch of images.

        Args:
            images (`ImageInput`):
                A list of images to preprocess.
            image_size (`int`, *optional*, defaults to `self.image_size`):
                Resize to image size
            image_num_channels (`int`, *optional*, defaults to `self.image_num_channels`):
                Number of image channels.
            image_mean (`float` or `List[float]`, *optional*, defaults to `EVOLLA_STANDARD_MEAN`):
                Mean to use if normalizing the image. This is a float or list of floats the length of the number of
                channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can
                be overridden by the `image_mean` parameter in the `preprocess` method.
            image_std (`float` or `List[float]`, *optional*, defaults to `EVOLLA_STANDARD_STD`):
                Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
                number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess`
                method. Can be overridden by the `image_std` parameter in the `preprocess` method.
            transform (`Callable`, *optional*, defaults to `None`):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple transforms. If `None` - an inference mode is
                assumed - and then a preset of inference-specific transforms will be applied to the images

        Returns:
            a PyTorch tensor of the processed images

        """
        image_size = image_size if image_size is not None else self.image_size
        image_num_channels = image_num_channels if image_num_channels is not None else self.image_num_channels
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = (image_size, image_size)

        if isinstance(images, list) and len(images) == 0:
            return []

        images = make_list_of_images(images)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        # For training a user needs to pass their own set of transforms as a Callable.
        # For reference this is what was used in the original EVOLLA training:
        # transform = transforms.Compose([
        #     convert_to_rgb,
        #     transforms.RandomResizedCrop((size, size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])
        if transform is not None:
            if not is_torch_available():
                raise ImportError("To pass in `transform` torch must be installed")
            import torch

            images = [transform(x) for x in images]
            return torch.stack(images)

        # for inference we do the exact transforms that were used to train EVOLLA
        images = [convert_to_rgb(x) for x in images]
        # further transforms expect numpy arrays
        images = [to_numpy_array(x) for x in images]
        images = [resize(x, size, resample=PILImageResampling.BICUBIC) for x in images]
        images = [self.rescale(image=image, scale=1 / 255) for image in images]
        images = [self.normalize(x, mean=image_mean, std=image_std) for x in images]
        images = [to_channel_dimension_format(x, ChannelDimension.FIRST) for x in images]
        images = BatchFeature(data={"pixel_values": images}, tensor_type=return_tensors)["pixel_values"]

        return images


__all__ = ["EvollaProteinProcessor"]
