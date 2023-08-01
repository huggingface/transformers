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
"""Processor class for KOSMOS-2."""

import math
import numpy as np
import re
from typing import List, Optional, Tuple, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, is_batched
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from ...utils import TensorType
from ...utils import is_tf_available, is_torch_available

import copy

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf


BboxInput = Union[
    List[Tuple[int, int]],
    List[Tuple[float, float, float, float]],
    List[List[Tuple[int, int]]],
    List[List[Tuple[float, float, float]]]
]


class Kosmos2Processor(ProcessorMixin):
    r"""
    Constructs an KOSMOS-2 processor which wraps a CLIP image processor and a KOSMOS-2 tokenizer into a single
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
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, List[TextInput]] = None,
        bboxes: BboxInput = None,
        num_image_tokens: Optional[int] = 64,
        first_image_token_id: Optional[int] = None,
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
            raise ValueError("You have to specify at least `text`.")

        text = self.preprocess_text(text, images, bboxes, num_image_tokens=num_image_tokens)

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
        encoding.update(text_encoding)

        if images is not None:
            image_encoding = self.image_processor(images, return_tensors=return_tensors)
            encoding.update(image_encoding)

            # Use the id of the first token after <unk>
            if first_image_token_id is None:
                first_image_token_id = self.tokenizer.unk_token_id + 1

            # To see if we need one more `0` (for `<s>`) at the beginning of `img_attn_mask`.
            with_bos = add_special_tokens

            # The first (actual) `<image>` token is always at the 1st or 2nd place (after `<s>` if any). Here we look
            # for the second `<image>` token (which indicate the first image token).
            start_index = int(with_bos) + 1

            if return_tensors:

                # change the ids for the fake `<image>` tokens in `input_ids`
                input_ids = np.array(encoding["input_ids"])
                input_ids[:, start_index:(start_index + num_image_tokens)] = np.arange(first_image_token_id, first_image_token_id + num_image_tokens)

                batch_size, seq_len = input_ids.shape[:2]
                img_attn_mask = []
                if with_bos:
                    # for `<s>`
                    img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # for `<image>` (the real one)
                img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # for image tokens
                img_attn_mask.append(np.ones(shape=(batch_size, 64), dtype=np.int64))
                # for `</image>`
                img_attn_mask.append(np.zeros(shape=(batch_size, 1), dtype=np.int64))
                # trailing part (which are not related to the image)
                seq_len -= (int(with_bos) + 1 + num_image_tokens + 1)
                img_attn_mask.append(np.zeros(shape=(batch_size, seq_len), dtype=np.int64))

                # concatenate along the sequence dimension
                img_attn_mask = np.concatenate(img_attn_mask, axis=1)

                # to the target tensor type
                if return_tensors == "pt":
                    input_ids = torch.from_numpy(input_ids)
                    img_attn_mask = torch.from_numpy(img_attn_mask)
                elif return_tensors == "tf":
                    input_ids = tf.convert_to_tensor(input_ids)
                    img_attn_mask = tf.convert_to_tensor(img_attn_mask)

                encoding["input_ids"] = input_ids
                encoding["img_attn_mask"] = img_attn_mask

            else:
                # Add `img_attn_mask`: the leading and trailing `0` are for `boi` and `eoi` tokens. The `1` indicates
                # the places of image tokens.
                image_token_ids = list(range(first_image_token_id, first_image_token_id + num_image_tokens))
                base_img_attn_mask = [0] + [1] * num_image_tokens + [0]

                # loop over `encoding["input_ids"]`
                input_ids = []
                img_attn_mask = []
                all_input_ids = encoding["input_ids"]
                # not batched -> (changed to) batch of size 1
                if isinstance(text, str):
                    all_input_ids = [all_input_ids]
                for text_ids in all_input_ids:
                    # change the ids for the fake `<image>` tokens in `input_ids`
                    text_ids = text_ids[:start_index] + image_token_ids + text_ids[start_index + num_image_tokens:]
                    input_ids.append(text_ids)

                    mask = copy.copy(base_img_attn_mask)
                    if with_bos:
                        # for `<s>`
                        mask = [0] + mask
                    # trailing part (which are not related to the image)
                    mask += [0] * (len(text_ids) - len(mask))
                    img_attn_mask.append(mask)

                # un-batch if necessary
                if isinstance(text, str):
                    input_ids = input_ids[0]
                    img_attn_mask = img_attn_mask[0]

                encoding["input_ids"] = input_ids
                encoding["img_attn_mask"] = img_attn_mask

        return encoding

    def preprocess_text(
        self,
        texts: Union[TextInput, List[TextInput]],
        images: ImageInput = None,
        bboxes: BboxInput = None,
        num_image_tokens: Optional[int] = 64,
    ) -> Union[str, List[str]]:
        """Add image and bounding box information to `texts` as image and patch index tokens.

        Args:
            texts (`Union[TextInput, List[TextInput]]`): The texts to be processed.
            images (`ImageInput`, *optional*): The images associated to `texts`.
            bboxes (`Union[List[Tuple[int]], List[Tuple[float]], List[List[Tuple[int]]], List[List[Tuple[float]]]]`, *optional*): The bounding bboxes associated to `texts`.
            num_image_tokens (`int`, *optional*, defaults to 64): The number of image tokens (used as latent queries). This should corresponds to the `latent_query_num` attribute in `Kosmos2Config`.

        Returns:
            `Union[TextInput, List[TextInput]]`: The processed texts with image and patch index tokens.
        """
        # These are fake `<image>` tokens enclosed between (the actual) `<image>` token and `</image>`.
        img_tokens = ["<image>"] * num_image_tokens
        img_info = " ".join(["<image>"] + img_tokens + ["</image>"])

        def check_bboxes_for_single_text(bboxes):
            """
            Check `bboxes` for a single text example. It could be
                - `None`: no bounding box associated to a text.
                - A list with each element being the bounding boxes associated to one `<phrase> ... </phrase>` pair
                  found in a text. This could be:
                      - `None`: no bounding box associated to a `<phrase> ... </phrase>` pair.
                      - A tuple of 2 integers: A single bounding box specified by patch indices.
                      - A tuple of 4 float point number: A single bounding box specified by (normalized) coordinates.
                      - A list containing the above 2 tuple types: Multiple bounding boxes for a
                       `<phrase> ... </phrase>` pair.
            """
            if bboxes is None:
                return
            elif not isinstance(bboxes, list):
                raise ValueError("`bboxes` (for a single text example) should be `None` or a list.")

            # `bbox` is the bounding boxes for a single <phrase> </phrase> pair
            for bbox in bboxes:
                if bbox is None:
                    continue
                elif not isinstance(bbox, list):
                    bbox = [bbox]
                for elt in bbox:
                    if not isinstance(elt, tuple) or not ((len(elt) == 2 and all(isinstance(x, int) for x in elt)) or (len(elt) == 4 and all(isinstance(x, float) for x in elt))):
                        raise ValueError(
                            "Each element in `bboxes` (for a single text example) should be `None`, a tuple containing "
                            "2 integers or 4 float point numbers, or a list containing such tuples. Also "
                            "make sure the arguments `texts` and `bboxes` passed to `preprocess_text` are both in "
                            "batches or both for a single example."
                    )

        def preprocess_single(text, image, bboxes):
            if image is not None:
                # Add `<image> ... (fake) image tokens ... </image>`
                text = f"{img_info} {text}"

                # Add `<object> <patch_idx_xxxx> <patch_idx_yyy> </object>` after `<phrase> phrase text </phrase>`
                text = self._insert_patch_index_tokens(text, bboxes)
                text = self._add_remove_spaces_around_tag_tokens(text)

            return text

        # make batch to simplify processing logic
        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]

        if images is None:
            images = [None] * len(texts)
        elif not is_batched(images):
            images = [images]
        if len(texts) != len(images):
            raise ValueError(f"The number of examples in `texts` and `images` should be the same. Got {len(texts)} v.s. {len(images)} instead.")

        if not batched:
            check_bboxes_for_single_text(bboxes)
            bboxes = [bboxes]
        elif bboxes is not None:
            if not isinstance(bboxes, list):
                raise ValueError("`bboxes` should be `None` or a list (as a batch) when `texts` is passed as a batch.")
            for x in bboxes:
                check_bboxes_for_single_text(x)
        else:
            bboxes = [None] * len(texts)

        if len(bboxes) != len(texts):
            raise ValueError(f"The number of examples in `texts` and `bboxes` should be the same. Got {len(texts)} v.s. {len(bboxes)} instead.")

        result = [preprocess_single(text, image, bbox) for text, image, bbox in zip(texts, images, bboxes)]
        # un-batch if necessary
        if not batched:
            result = result[0]

        return result

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

    def _insert_patch_index_tokens(self, text: str, bboxes: Union[List[Tuple[int]], List[Tuple[float]]]) -> str:
        if bboxes is None or len(bboxes) == 0:
            return text

        matched_phrases = list(re.finditer(r"<phrase>.+?</phrase>", string=text))
        if len(matched_phrases) != len(bboxes):
            raise ValueError(f"The number of elements in `bboxes` should be the same as the number of `<phrase> ... </phrase>` pairs in `text`. Got {len(matched_phrases)} v.s. {len(bboxes)} instead.")

        # insert object's patch index tokens
        # the found `<phrase> ... </phrase>` pairs.
        curr_pos = 0
        buffer = []
        for matched, bbox in zip(matched_phrases, bboxes):
            _, end = matched.span()
            buffer.append(text[curr_pos:end])
            curr_pos = end
            # A phrase without bbox
            if bbox is None:
                continue
            # A phrase with a single bbox
            if isinstance(bbox, tuple):
                bbox = [bbox]
            patch_index_strings = []
            # A phrase could have multiple bboxes
            for box in bbox:
                patch_index_1, patch_index_2 = self._convert_bbox_to_patch_index_tokens(box)
                patch_index_strings.append(f"{patch_index_1} {patch_index_2}")
            position_str = " </delimiter_of_multi_objects/> ".join(patch_index_strings)
            buffer.append(f"<object> {position_str} </object>")
        # remaining
        if curr_pos < len(text):
            buffer.append(text[curr_pos:])

        text = "".join(buffer)
        return text

    def _convert_bbox_to_patch_index_tokens(
        self, bbox: Union[Tuple[int, int], Tuple[float, float, float, float]]
    ) -> Tuple[str, str]:
        # already computed patch indices
        if len(bbox) == 2:
            idx_1, idx_2 = bbox
        # bbox specified with (normalized) coordinates
        else:
            # use `self.tokenizer` to get `num_patches_per_side`
            num_patches_per_side = int(math.sqrt(self.tokenizer.num_patch_index_tokens))
            idx_1, idx_2 = coordinate_to_patch_index(bbox, num_patches_per_side)

        token_1 = f"<patch_index_{str(idx_1).zfill(4)}>"
        token_2 = f"<patch_index_{str(idx_2).zfill(4)}>"

        return token_1, token_2

    def _add_remove_spaces_around_tag_tokens(self, text):
        """
        Remove spaces before tag tokens (e.g. `<x>`). Also ensure a space after a tag token, if it is not followed by
        another tag token (this is not technically necessary, but good for a standard/consistent format). This avoids
        the inconsistency of tokenization results between kosmos-2 slow and fast tokenizers.
        """

        tag_tokens = set(self.tokenizer.tag_tokens + [f"<patch_index_{str(x).zfill(4)}>" for x in range(self.tokenizer.num_patch_index_tokens)])
        pattern = "|".join(tag_tokens)
        splits = re.split(rf"({pattern})", text)

        output = ""
        prev_str_in_targets = False
        for split in splits:
            if split in tag_tokens:
                prev_str_in_targets = True
                output = output.rstrip() + split
            else:
                # we don't need to ensure a space before a normal token that is after a tag token. But having it and
                # keeps a standard format is good anyway.
                if prev_str_in_targets and not split.startswith(" "):
                    output += " " + split
                else:
                    output += split
                prev_str_in_targets = False

        return output


def coordinate_to_patch_index(bbox: Tuple[float, float, float, float], num_patches_per_side: int) -> Tuple[int, int]:
    """Convert a bounding box to a pair of patch indices.

    Args:
        bbox (`Tuple[float, float, float, float]`):
            The 4 coordinates of the bounding box, with the format being (x1, y1, x2, y2) specifying the upper-left
            and lower-right corners of the box. It should have x2 > x1 and  y1 > y2.
        num_patches_per_side (`int`): the number of patches along each side.

    Returns:
        `Tuple[int, int]`: A pair of patch indices.
    """
    (x1, y1, x2, y2) = bbox

    ul_x = math.floor(x1 * num_patches_per_side)
    ul_y = math.floor(y1 * num_patches_per_side)

    lr_x = math.ceil(x2 * num_patches_per_side - 1)
    lr_y = math.ceil(y2 * num_patches_per_side - 1)

    ul_idx = ul_y * num_patches_per_side + ul_x
    lr_idx = lr_y * num_patches_per_side + lr_x

    return ul_idx, lr_idx


# copied from https://github.com/microsoft/unilm/blob/97e4923e97d3ee10b57e97013556e3fd0d207a9b/kosmos-2/demo/decode_string.py#L35C1-L75C38
def patch_index_to_coordinate(ul_idx: int, lr_idx: int, num_patches_per_side: int):
    """
    Given a grid of length `num_patches_per_side` and the indices of the upper-left and lower-right corners of a
    bounding box, returns the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).

    Args:
        ul_idx (`int`): the index of the grid cell that corresponds to the upper-left corner of the bounding box.
        lr_idx (`int`): the index of the grid cell that corresponds to the lower-right corner of the bounding box.
        num_patches_per_side (`int`): the number of patches along each side.

    Returns:
        `Tuple[float]`: the normalized coordinates of the bounding box, in the form (x1, y1, x2, y2).
    """
    # Compute the size of each cell in the grid
    cell_size = 1.0 / num_patches_per_side

    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box
    ul_x = ul_idx % num_patches_per_side
    ul_y = ul_idx // num_patches_per_side

    lr_x = lr_idx % num_patches_per_side
    lr_y = lr_idx // num_patches_per_side

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

    return x1, y1, x2, y2
