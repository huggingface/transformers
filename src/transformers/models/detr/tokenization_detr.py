# coding=utf-8
# Copyright 2021 The Facebook Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization class for DETR."""

import json
import os
from itertools import groupby
from typing import Dict, List, Optional, Tuple, Union
import random

import numpy as np
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms as T
import PIL

from ...file_utils import add_end_docstrings
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, TensorType
from ...utils import logging


logger = logging.get_logger(__name__)


DETR_KWARGS_DOCSTRING = r"""
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            max_length (:obj:`int`, `optional`):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum
                length is required by one of the truncation/padding parameters. If the model has no specific maximum
                input length (like XLNet) truncation/padding to a maximum length will be deactivated.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            verbose (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to print more information and warnings.
"""

## BELOW: utilities copied from 
# https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/util/misc.py


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    """
    Data type that handles different types of inputs (either list of images or list of sequences),
    and computes the padded output (with masking).
    """
    def __init__(self, tensors, mask: Optional[Tensor]):
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


def nested_tensor_from_tensor_list(tensor_list: Union[List[Tensor], torch.Tensor]):
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
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('Not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


## Image + target transformations for object detection
## Taken from https://github.com/facebookresearch/detr/blob/master/datasets/transforms.py

# this: extra transform, based on https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/datasets/coco.py#L21
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        #image_id = target["image_id"]
        #image_id = torch.tensor([image_id])

        #anno = target["annotations"]

        #anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        #boxes = [obj["bbox"] for obj in anno]
        if target is not None:
            boxes = target["boxes"]
            # guard against no boxes via resizing
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

            #classes = [obj["category_id"] for obj in anno]
            #classes = torch.tensor(classes, dtype=torch.int64)

            if self.return_masks:
                segmentations = [obj["segmentation"] for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)

            # keypoints = None
            # if anno and "keypoints" in anno[0]:
            #     keypoints = [obj["keypoints"] for obj in anno]
            #     keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            #     num_keypoints = keypoints.shape[0]
            #     if num_keypoints:
            #         keypoints = keypoints.view(num_keypoints, -1, 3)

            # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            # boxes = boxes[keep]
            # classes = classes[keep]
            # if self.return_masks:
            #     masks = masks[keep]
            # if keypoints is not None:
            #     keypoints = keypoints[keep]

            target = {}
            target["boxes"] = boxes
            # target["labels"] = classes
            # if self.return_masks:
            #     target["masks"] = masks
            # target["image_id"] = image_id
            # if keypoints is not None:
            #     target["keypoints"] = keypoints

            # # for conversion to coco api
            # area = torch.tensor([obj["area"] for obj in anno])
            # iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
            # target["area"] = area[keep]
            # target["iscrowd"] = iscrowd[keep]

            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])
        
            return image, target
        return image, None


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, image, target=None):
        return resize(image, target, self.size, self.max_size)

# copied from https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/util/box_ops.py
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class DetrTokenizer(PreTrainedTokenizer):
    """
    Constructs a DETR tokenizer.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains some of the main methods.
    Users should refer to the superclass for more information regarding such methods.

    Args:
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sentence token.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sentence token.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`"|"`):
            The token used for defining the end of a word.
        **kwargs
            Additional keyword arguments passed along to :class:`~transformers.PreTrainedTokenizer`
    """

    model_input_names = ["input_values"]

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            do_lower_case=do_lower_case,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )
        self._word_delimiter_token = word_delimiter_token
        self.do_lower_case = do_lower_case

    @add_end_docstrings(DETR_KWARGS_DOCSTRING)
    def __call__(
        self,
        images: Union[PIL.Image.Image, np.ndarray, torch.Tensor, List[PIL.Image.Image], List[np.ndarray], List[torch.Tensor]],
        annotations: Optional[Union[Dict, List[Dict]]] = None,
        padding: Union[bool, str] = True,
        return_mask: Union[bool, str] = True,
        resize:  Optional[bool] = True,
        size: Optional[int] = 800,
        max_size: Optional[int] = 1333,
        normalize: Optional[bool] = True,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Main method to prepare for the model one or several image(s). This includes resizing, normalization and padding up to 
        the largest image in a batch while creating a pixel mask for each image indicating which pixels are real and which are padding. 

        Args:
            images (:obj:`PIL.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`,
            :obj:`List[PIL.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, numpy array or a Torch tensor.
            annotations (:obj:`Dict`, :obj:`List[Dict]`):
                The annotations as either a Python dictionary (in case of a single image) or a list of Python dictionaries (in case
                of a batch of images). Keys include "boxes", "labels", "area" and "masks". 
            resize (:obj:`bool`, `optional`, defaults to :obj:`True`): 
                Whether to resize images (and annotations) to a certain :obj:`size`.
            size (:obj:`int`, `optional`, defaults to :obj:`800`): 
                Resize the input image to the given size. Only has an effect if :obj:`resize` is set to :obj:`True`. 
            max_size (:obj:`int`, `optional`, defaults to :obj:`1333`): 
                The largest size an image dimension can have (otherwise it's capped). In COCO, the authors of DETR used a :obj:`max_size` of 1333.
            normalize (:obj:`bool`, `optional`, defaults to :obj:`True`): 
                Whether to apply standard ImageNet mean/std normalization of images. 
        """

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (PIL.Image.Image, np.ndarray, torch.Tensor)))
        )

        # step 1: make images a list of PIL images no matter what
        if is_batched:
            if isinstance(images[0], np.ndarray):
                images = [Image.fromarray(image) for image in images]
            elif isinstance(images[0], torch.Tensor):
                images = [T.ToPILImage()(image).convert("RGB") for image in images]
            if annotations is not None:
                assert len(images) == len(annotations)
        else:
            if isinstance(images, PIL.Image.Image):
                images = [images]
            if annotations is not None:
                annotations = [annotations]

        # step 2: define transformations (resizing + normalization) 
        transformations = [ConvertCocoPolysToMask(),]
        if resize and size is not None: 
            transformations.append(Resize(size=size, max_size=max_size))
        if normalize:
            normalization = Compose([
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            transformations.append(normalization)
        transforms = Compose(transformations)

        # step 3: apply transformations to both images and annotations
        transformed_images = []
        transformed_annotations = []
        if annotations is not None:
            for image, annotation in zip(images, annotations):
                image, annotation = transforms(image, annotation)
                transformed_images.append(image)
                transformed_annotations.append(annotation)
        else:
            transformed_images = [transforms(image, None)[0] for image in images]
        
        # step 4: create NestedTensor which takes care of padding the pixels up to biggest image
        # and creation of mask. We don't need the transformed targets for this
        samples = nested_tensor_from_tensor_list(transformed_images)

        # return as BatchEncoding
        data = {'pixel_values': samples.tensors, 'pixel_mask': samples.mask} 
        
        if annotations is not None:
            data['labels'] = transformed_annotations

        encoded_inputs = BatchEncoding(data=data)

        return encoded_inputs
    
    @property
    def vocab_size(self) -> int:
        return len(self.decoder)

    def get_vocab(self) -> Dict:
        return dict(self.encoder, **self.added_tokens_encoder)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        grouped_tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, grouped_tokens))

        # replace delimiter token
        string = "".join([" " if token == self.word_delimiter_token else token for token in filtered_tokens]).strip()

        if self.do_lower_case:
            string = string.lower()
        return string

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        special _decode function is needed for DETRTokenizer because added tokens should be treated exactly the
        same as tokens of the base vocabulary and therefore the function `convert_tokens_to_string` has to be called on
        the whole token list and not individually on added tokens
        """
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(result)

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text