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
"""Image processor class for ViT."""

import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms as T

from typing import Optional, Union, List

from ...file_utils import PaddingStrategy, TensorType
from ...image_processor_utils import BatchImages, PreTrainedImageProcessor
from ...utils import logging


logger = logging.get_logger(__name__)

## BELOW: utilities copied from
## https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/util/misc.py


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """
    Data type that handles different types of inputs (either list of images or list of sequences), and computes the
    padded output (with masking).
    """

    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: Union[List[torch.Tensor], torch.Tensor]):
    # TODO make this more n
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = True
    else:
        raise ValueError("Not supported")
    return NestedTensor(tensor, mask)


class ViTImageProcessor(PreTrainedImageProcessor):
    r"""
    Constructs a ViT image processor. This image processor inherits from
    :class:`~transformers.PreTrainedImageProcessor` which contains most of the main methods. Users should refer to this
    superclass for more information regarding those methods.
    Args:
        image_mean (:obj:`int`, defaults to [0.485, 0.456, 0.406]):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (:obj:`int`, defaults to [0.229, 0.224, 0.225]):
            The sequence of standard deviations for each channel, to be used when normalizing images.
        padding_value (:obj:`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        return_attention_mask (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not :meth:`~transformers.ViTImageProcessor.__call__` should return :obj:`attention_mask`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int`, `optional`, defaults to :obj:`224`):
            Resize the input to the given size. Only has an effect if :obj:`resize` is set to :obj:`True`.
    """

    model_input_names = ["pixel_values", "attention_mask"]

    def __init__(
        self,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        padding_value=0.0,
        return_attention_mask=True,
        do_normalize=True,
        do_resize=True,
        size=224,
        **kwargs
    ):
        super().__init__(image_mean=image_mean, image_std=image_std, padding_value=padding_value, **kwargs)
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.size = size

    def __call__(
        self,
        images: Union[
            PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray], List[torch.Tensor]
        ],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_resolution: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        verbose: bool = True,
        **kwargs
    ) -> BatchImages:
        """
        Main method to prepare for the model one or several image(s).
        Args:
            images (:obj:`PIL.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, numpy array or a Torch
                tensor.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:
                * :obj:`True` or :obj:`'biggest'`: Pad to the biggest image in the batch (or no padding if only a
                  single image is provided).
                * :obj:`'max_resolution'`: Pad to a maximum resolution specified with the argument
                  :obj:`max_resolution` or to the maximum acceptable input resolution for the model if that argument is
                  not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with images of
                  different resolutions).
            max_resolution (:obj:`int`, `optional`):
                Controls the maximum resolution to use by one of the truncation/padding parameters. If left unset or
                set to :obj:`None`, this will use the predefined model maximum resolution if a maximum resolution is
                required by one of the truncation/padding parameters. If the model has no specific maximum input
                resolution, truncation/padding to a maximum resolution will be deactivated.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_attention_mask (:obj:`bool`, `optional`):
                Whether to return the pixel mask. If left to the default, will return the pixel mask according
                to the specific image processor's default.
                `What are pixel masks? <../glossary.html#attention-mask>`__
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python floats. Acceptable values are:
                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
        """
        # Input type checking for clearer error
        assert (
            isinstance(images, PIL.Image.Image)
            or isinstance(images, np.ndarray)
            or isinstance(images, torch.Tensor)
            or (
                (
                    isinstance(images, (list, tuple))
                    and (
                        len(images) == 0
                        or (
                            isinstance(images[0], PIL.Image.Image)
                            or isinstance(images[0], np.ndarray)
                            or isinstance(images[0], torch.Tensor)
                        )
                    )
                )
            )
        ), (
            "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
            "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
        )

        is_batched = bool(
            isinstance(images, (list, tuple)) and (isinstance(images[0], (PIL.Image.Image, np.ndarray, torch.Tensor)))
        )

        # step 1: make images a list of PIL images no matter what
        if is_batched:
            if isinstance(images[0], np.ndarray):
                images = [Image.fromarray(image).convert("RGB") for image in images]
            elif isinstance(images[0], torch.Tensor):
                images = [T.ToPILImage()(image).convert("RGB") for image in images]
        else:
            if isinstance(images, np.ndarray):
                images = [Image.fromarray(images).convert("RGB")]
            elif isinstance(images, torch.Tensor):
                images = [T.ToPILImage()(images).convert("RGB")]
            else:
                images = [images]

        # step 2: define transformations (resizing + normalization)
        transformations = []
        if self.do_resize and self.size is not None:
            transformations.append(T.Resize(size=(self.size, self.size)))
        if self.do_normalize:
            normalization = T.Compose([T.ToTensor(), T.Normalize(self.image_mean, self.image_std)])
            transformations.append(normalization)
        transforms = T.Compose(transformations)

        # step 3: apply transformations to images
        transformed_images = [transforms(image) for image in images]

        # step 4: TO DO: replace by self.pad (which is defined in image_processor_utils.py), which should 
        # take care of padding, creation of attention mask, return_tensors type
        samples = nested_tensor_from_tensor_list(transformed_images)

        # return as BatchImages
        data = {"pixel_values": samples.tensors, "attention_mask": samples.mask}

        encoded_inputs = BatchImages(data=data)

        return encoded_inputs