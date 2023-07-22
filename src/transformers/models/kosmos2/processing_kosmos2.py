# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Processor class for Kosmos-2. Largely copy of Blip2Processor with addition of a tokenizer for the Q-Former.
"""

import os
from typing import List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType
from ..auto import AutoTokenizer
from ...utils import is_tf_available, is_torch_available

import copy

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


class Kosmos2Processor(ProcessorMixin):
    r"""
    Constructs an Kosmos-2 processor which wraps a CLIP image processor and a Kosmos-2 tokenizer into a single
    processor.

    [`Kosmos2Processor`] offers all the functionalities of [`CLIPImageProcessor`] and [`Kosmos2TokenizerFast`]. See the
    docstring of [`~Kosmos2Processor.__call__`] and [`~Kosmos2Processor.decode`] for more information.

    Args:
        image_processor (`CLIPImageProcessor`):
            An instance of [`CLIPImageProcessor`]. The image processor is a required input.
        tokenizer (`Kosmos2TokenizerFast`):
            An instance of ['Kosmos2TokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("Kosmos2Tokenizer", "Kosmos2TokenizerFast")

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        bbox: Union[None] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        This method uses [`CLIPImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`Kosmos2TokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        if text is None:
            raise ValueError("You have to specify at least text.")

        text = self.preprocess_text(text, images, bbox)

        encoding = BatchFeature()

        text_encoding = self.tokenizer(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_token_type_ids=return_token_type_ids,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
        encoding["decoder_input_ids"] = text_encoding["input_ids"]
        encoding["decoder_attention_mask"] = text_encoding["attention_mask"]

        if images is not None:
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding["pixel_values"] = image_encoding["pixel_values"]

            # Add `img_attn_mask`

            # The leading and trailing `0` are for <boi> and <eoi> tokens. The `1`s indicate the places of image tokens.
            image_token_ids = list(range(4, 4 + 64))
            base_img_attn_mask = [0] + [1] * 64 + [0]
            # To see if we need one more `0` for `img_attn_mask` at the beginning
            with_bos = add_special_tokens

            # The start of 2nd <image>
            start_index = int(with_bos) + 1

            if return_tensors:

                import numpy as np

                # change <image> ids in `decoder_input_ids`
                decoder_input_ids = np.array(encoding["decoder_input_ids"])
                decoder_input_ids[:, start_index:(start_index + 64)] = np.arange(4, 4 + 64)

                batch_size, seq_len = decoder_input_ids.shape[:2]
                img_attn_mask = []
                if with_bos:
                    img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # <boi>
                img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # image tokens
                img_attn_mask.append(np.ones(shape=(batch_size, 64), dtype=np.int64))
                # <eoi>
                img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # trailing part
                seq_len -= (int(with_bos) + 1 + 64 + 1)
                img_attn_mask.append(np.zeros(shape=(batch_size, seq_len), dtype=np.int64))

                # concatenate along the sequence dimension
                img_attn_mask = np.concatenate(img_attn_mask, axis=1)

                # to the target tensor type
                if return_tensors == "pt":
                    decoder_input_ids = torch.from_numpy(decoder_input_ids)
                    img_attn_mask = torch.from_numpy(img_attn_mask)
                elif return_tensors == "tf":
                    decoder_input_ids = tf.convert_to_tensor(decoder_input_ids)
                    img_attn_mask = tf.convert_to_tensor(img_attn_mask)

                encoding["decoder_input_ids"] = decoder_input_ids
                encoding["img_attn_mask"] = img_attn_mask

            else:
                # loop over `text_encoding["input_ids"]`
                input_ids = []
                img_attn_mask = []
                all_input_ids = encoding["decoder_input_ids"]
                # not batched -> batch of size 1
                if isinstance(text, str):
                    all_input_ids = [all_input_ids]
                for text_ids in all_input_ids:
                    text_ids = text_ids[:start_index] + image_token_ids + text_ids[start_index + 64:]
                    input_ids.append(text_ids)
                    mask = copy.copy(base_img_attn_mask)
                    if with_bos:
                        mask = [0] + mask
                    mask += [0] * (len(text_ids) - len(mask))
                    img_attn_mask.append(mask)

                # un-batch
                if isinstance(text, str):
                    input_ids = input_ids[0]
                    img_attn_mask = img_attn_mask[0]
                encoding["decoder_input_ids"] = input_ids
                encoding["img_attn_mask"] = img_attn_mask

        return encoding

    def preprocess_text(self, texts, images=None, bboxes=None):

        if images is None:
            return texts

        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]

        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images)

        if bboxes is not None:
            if not isinstance(bboxes, list):
                # A tuple of 2 elements: (float, float)
                bboxes = [bboxes]
            elif bboxes[0] is None or isinstance(bboxes[0], tuple):
                bboxes = [bboxes]
            assert len(texts) == len(bboxes)
        else:
            bboxes = [None] * len(texts)

        # These are same as text tokens, but they will be enclosed between <image> and </image>.
        img_tokens = ["<image>"] * 64
        img_info = " ".join(["<image>"] + img_tokens + ["</image>"])

        def preprocess_single(text, image, bbox):

            # Add <image> tag </image>
            if image is not None:
                text = f"{img_info} {text}"

                # Add `<object> <patch_idx_xxxx> <patch_idx_yyy> </object>` after `<phrase> text </phrase>`
                if bbox is not None and len(bbox) > 0:
                    text = self.insert_patch_index_tokens(text, image, bbox)

            return text

        result = [preprocess_single(text, image, bbox) for text, image, bbox in zip(texts, images, bboxes)]
        if not batched:
            result = result[0]

        return result

    def insert_patch_index_tokens(self, text, image, bboxes):

        buffer = []

        # TODO: add a check of equal number of bboxes and <phrase> </phrase>
        pos = text.find("</phrase>")
        while pos > -1:
            buffer.append(text[:pos + len("</phrase>")])
            # A <phrase> </phrase> without any associated bbox
            bbox = bboxes.pop()
            if bbox is not None:
                buffer.append(self.convert_bbox_to_patch_index_tokens(image, bbox))
            text = text[pos + len("</phrase>"):]
            pos = text.find("</phrase>")
        # remaining
        if text:
            buffer.append(text)

        text = " ".join(buffer)

        return text

    def convert_bbox_to_patch_index_tokens(self, image, bbox):

        if not isinstance(bbox, tuple):
            raise ValueError(f"`bbox` needs to be a tuple. Got {type(bbox)} instead.")
        elif len(bbox) != 2:
            raise ValueError(f"`bbox` needs to be a tuple of 2 elements. Got a tuple of {len(bbox)} elements instead.")

        bbox_0, bbox_1 = bbox

        if isinstance(bbox_0, float) and isinstance(bbox_1, float):
            bbox_0 = None
            bbox_1 = None
        elif isinstance(bbox_0, int) and isinstance(bbox_1, int):
            bbox_0 = f"<patch_index_{str(bbox_0).zfill(4)}>"
            bbox_1 = f"<patch_index_{str(bbox_1).zfill(4)}>"
        elif isinstance(bbox_0, str) and isinstance(bbox_1, str):
            pass
        else:
            raise ValueError(f"`bbox` needs to be a tuple of 2 elements of the same type `float`, `int` or `str`. Got `({type(bbox_0)}, {type(bbox_1)})` instead.")

        return f"<object> {bbox_0} {bbox_1} </object>"

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def bbox_to_patch_index(self, bbox, P=32):
        # TODO: un-normalized version

        # The assumption is: x2 > x1 and  y1 > y2
        (x1, y1, x2, y2) = bbox

        import math
        ul_x = math.floor(x1 * P)
        ul_y = math.floor(y1 * P)

        lr_x = math.floor(x2 * P - 1)
        lr_y = math.floor(y2 * P - 1)

        ul_idx = ul_y * P + ul_x
        lr_idx = lr_y * P + lr_x

        return ul_idx, lr_idx

    # copied from https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L35C1-L75C38
    # TODO: clean up + un-normalized version
    def patch_index_to_bbox(self, ul_idx, lr_idx, P=32):
        """
        Given a grid of length P and the indices of the upper-left and lower-right corners of a bounding box,
        returns the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2].

        Args:
        - P (int): the length of the grid
        - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box
        - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box

        Returns:
        - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]
        """
        # Compute the size of each cell in the grid
        cell_size = 1.0 / P

        # Compute the x and y indices of the upper-left and lower-right corners of the bounding box
        ul_x = ul_idx % P
        ul_y = ul_idx // P

        lr_x = lr_idx % P
        lr_y = lr_idx // P

        # Compute the normalized coordinates of the bounding box
        if ul_idx == lr_idx:
            x1 = ul_x * cell_size
            y1 = ul_y * cell_size
            x2 = lr_x * cell_size + cell_size
            y2 = lr_y * cell_size + cell_size
        elif ul_x == lr_x or ul_y == lr_y:
            x1 = ul_x * cell_size
            y1 = ul_y * cell_size
            x2 = lr_x * cell_size + cell_size
            y2 = lr_y * cell_size + cell_size
        else:
            x1 = ul_x * cell_size + cell_size / 2
            y1 = ul_y * cell_size + cell_size / 2
            x2 = lr_x * cell_size + cell_size / 2
            y2 = lr_y * cell_size + cell_size / 2

        import numpy as np
        return np.array([x1, y1, x2, y2])
