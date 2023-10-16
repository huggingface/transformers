from __future__ import annotations
from .image_processing_fuyu import FuyuImageProcessor

from enum import IntEnum, auto
from typing import List, Sequence, TypeVar

import numpy as np
from typing import Callable, NamedTuple, Optional, Union

import einops
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.functional as TF  # type: ignore
from PIL import Image

"""Monza data processing."""
import math
from typing import List, NamedTuple, Tuple, Union, Optional

import torch

from transformers import AutoTokenizer
import re

from enum import IntEnum, auto
from typing import List, Sequence, TypeVar

import numpy as np


# TODO(fjord): Once we're using the new vocabulary, switch to reserved/prioritized tokens like <bbox> and <point>.
BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
POINT_OPEN_STRING = "<0x02>"  # <point>
POINT_CLOSE_STRING = "<0x03>"  # </point>

T = TypeVar("T")


class BatchInput(NamedTuple):
    """Batch input to be used in the model forward pass."""

    image_input: MonzaImageInput
    tokens: torch.Tensor
    position_ids: torch.Tensor

    def cpu(self) -> BatchInput:
        """move to cpu"""
        return BatchInput(
            image_input=self.image_input.cpu(),
            tokens=self.tokens.cpu(),
            position_ids=self.position_ids.cpu(),
        )


class MonzaImageInput(NamedTuple):
    """Image input to be used by MonzaModel.forward.

    images: A list of lists of tensors, one for each image in the batch. The outer list is over batch entries,
        the inner list is over subsequence entries. The tensors are of shape [c, h, w].
        Critical: only images that are actually present (as indicated by the image_present tensor) are included.
        All images in this list are expected to go through the image encoder and used for model input.
    image_patch_input_indices: A tensor of shape [b, s] where each entry is either -1 or the index of an image
        patch embedding that should be used in that position.
    """

    images: List[List[torch.Tensor]]
    image_patch_input_indices: torch.Tensor

    def cpu(self) -> MonzaImageInput:
        """Move to cpu"""
        images: List[List[torch.Tensor]] = []
        for image_batch in self.images:
            images.append([x.cpu() for x in image_batch])
        return MonzaImageInput(
            images=images,
            image_patch_input_indices=self.image_patch_input_indices.cpu(),
        )


class BatchTarget(NamedTuple):
    """Dataclass for batch target to be used in the model forward pass.

    image_patch_label_indices_per_batch: A tensor of shape [b, s] where each entry is either -1 or the index of an image
        patch embedding that should be predicted for that position.
        This is offset by 1 as compared to image_patch_input_indices in MonzaImageInput because these are targets.
    image_patch_label_indices_per_subsequence: Same as image_patch_label_indices_per_batch, but the indices are
        relative to the subsequence rather than the batch (i.e., the index counter is reset at each subsequence
        boundary). This is useful for reconstructing an individual image from predicted patches regardless of
        which subsequence it was in.
    subsequence_ids: Tensor corresponding to labels that indicates which input subsequence a given label token belongs
        to. This allows for recovering which part of a sequence came from which subsequence before packing.
        It also makes it clear which tokens belong to which image.
    """

    labels: torch.Tensor
    image_patch_label_indices_per_batch: torch.Tensor
    image_patch_label_indices_per_subsequence: torch.Tensor
    image_patch_labels: List[torch.Tensor]
    token_loss_mask: torch.Tensor
    token_types: torch.Tensor
    subsequence_ids: torch.Tensor
    dataset_task_source: torch.Tensor
    original_sequence_lengths: torch.Tensor
    sequence_is_truncated: torch.Tensor

    def cpu(self) -> BatchTarget:
        """move to cpu"""
        return BatchTarget(
            labels=self.labels.cpu(),
            image_patch_labels=[x.cpu() for x in self.image_patch_labels],
            image_patch_label_indices_per_batch=self.image_patch_label_indices_per_batch.cpu(),
            image_patch_label_indices_per_subsequence=self.image_patch_label_indices_per_subsequence.cpu(),
            token_loss_mask=self.token_loss_mask.cpu(),
            token_types=self.token_types.cpu(),
            subsequence_ids=self.subsequence_ids.cpu(),
            dataset_task_source=self.dataset_task_source.cpu(),
            original_sequence_lengths=self.original_sequence_lengths.cpu(),
            sequence_is_truncated=self.sequence_is_truncated.cpu(),
        )

# TODO(augustus): this should live somewhere different, since they'll want it in finetuning, etc


class TokenType(IntEnum):
    """The type of token - used for computing losses."""

    TAG = auto()
    TEXT = auto()
    NEWLINE = auto()
    LOCATION = auto()
    EOS = auto()
    PROMPT = auto()
    CAPTION_LABEL = auto()
    BOS = auto()
    IMAGE = auto()
    IMAGE_NEWLINE = auto()
    OTHER = auto()

    # The questions and answers in the bbox-center-to-token task.
    BBOX_CENTER_TO_TOKEN_QUERY = auto()
    BBOX_CENTER_TO_TOKEN_ANSWER = auto()

    # The questions and answers in the token-to-bbox-center task.
    TOKEN_TO_BBOX_CENTER_QUERY = auto()
    TOKEN_TO_BBOX_CENTER_ANSWER = auto()

    # For delimiters we don't take the loss on
    DELIMITER = auto()

    # Normal GPT-style tokens
    GPT_TOKEN = auto()

    # Finetuning
    FT_CONTEXT = auto()
    FT_HISTORY = auto()
    FT_OBSERVATION = auto()
    FT_COMMAND = auto()
    FT_ACTION = auto()
    FT_REWARD = auto()

    # For the openImages dataset
    BBOX_TO_POINTLABEL_QUERY = auto()
    BBOX_TO_POINTLABEL_ANSWER = auto()
    BBOX_TO_CLASS_LABEL_ANSWER = auto()

    # The questions and answers in the bbox-to-tokens task.
    BBOX_TO_TOKENS_QUERY = auto()
    BBOX_TO_TOKENS_ANSWER = auto()
    TOKENS_TO_BBOX_QUERY = auto()
    TOKENS_TO_BBOX_ANSWER = auto()

    VQA_QUESTION = auto()
    VQA_ANSWER = auto()

    CLASSIFY_IMAGE_QUERY = auto()
    CLASSIFY_IMAGE_ANSWER = auto()

    COUNT_IMAGE_QUERY = auto()
    COUNT_IMAGE_ANSWER = auto()

    GENERATED_CAPTION = auto()


CENTER_LOCATION_TOKEN_TYPES = [
    TokenType.BBOX_CENTER_TO_TOKEN_QUERY,
    TokenType.TOKEN_TO_BBOX_CENTER_ANSWER,
    TokenType.BBOX_TO_POINTLABEL_QUERY,
    TokenType.LOCATION,
]
BBOX_LOCATION_TOKEN_TYPES = [TokenType.BBOX_TO_TOKENS_QUERY, TokenType.TOKENS_TO_BBOX_ANSWER]
ANSWER_TOKEN_TYPES = [
    TokenType.CAPTION_LABEL,
    TokenType.BBOX_CENTER_TO_TOKEN_ANSWER,
    TokenType.TOKEN_TO_BBOX_CENTER_ANSWER,
    TokenType.FT_ACTION,
    TokenType.BBOX_TO_POINTLABEL_ANSWER,
    TokenType.BBOX_TO_CLASS_LABEL_ANSWER,
    TokenType.BBOX_TO_TOKENS_ANSWER,
    TokenType.TOKENS_TO_BBOX_ANSWER,
    TokenType.VQA_ANSWER,
    TokenType.CLASSIFY_IMAGE_ANSWER,
]

TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"


"""
from megatron.model import utils
from megatron.utils import print_once_per_data_rank
from multimodal.data.batch_enums import BatchInput, BatchTarget, MonzaImageInput
from multimodal.data.image_utils import get_num_patches, patchify_image
from multimodal.utils import next_greater_multiple
"""

processor = FuyuImageProcessor()
patchify_image = processor.patchify_image

get_num_patches = processor.get_num_patches


def next_greater_multiple(num: int, factor: int = 128) -> int:
    """Round num up to the next greatest multiple of factor."""
    return ((num + factor - 1) // factor) * factor


def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List[torch.Tensor],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> torch.Tensor:
    """Takes an unpacked stream of tokens (i.e. a list of tensors, one for each item in the batch) and does
    the required padding to create a single tensor for the batch of shape batch_size x new_seq_len.
    """

    assert len(all_bi_tokens_to_place) == batch_size
    assert len(full_unpacked_stream) == batch_size

    # Create padded tensors for the full batch.
    new_padded_tensor = torch.full(
        [batch_size, new_seq_len],
        fill_value=fill_value,
        dtype=full_unpacked_stream[0].dtype,
        device=full_unpacked_stream[0].device,
    )

    # Place each batch entry into the batch tensor.
    for bi in range(batch_size):
        tokens_to_place = all_bi_tokens_to_place[bi]
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset: tokens_to_place + offset]

    return new_padded_tensor


def construct_full_unpacked_stream(
    num_real_text_tokens: Union[List[List[int]], torch.Tensor],
    input_stream: torch.Tensor,
    image_tokens: List[List[torch.Tensor]],
    batch_size: int,
    num_sub_sequences: int,
) -> List[torch.Tensor]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per
    item in the batch. Returns a list of tensors, one for each item in the batch."""

    all_bi_stream = []

    for bi in range(batch_size):
        all_si_stream = []

        # First, construct full token stream (including image placeholder tokens) and loss mask for each subsequence
        # and append to lists. We use lists rather than tensors because each subsequence is variable-sized.
        for si in range(num_sub_sequences):
            image_adjustment = image_tokens[bi][si]
            breakpoint()
            si_stream = torch.cat([image_adjustment, input_stream[bi, si]], dim=0)
            num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[bi][si]

            all_si_stream.append(si_stream[:num_real_tokens])
        # Combine all subsequences for this batch entry. Still using a list because each batch entry is variable-sized.
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    return all_bi_stream


class ModelImageInput(NamedTuple):
    """Image inputs formatted to go into the model.

    images: A list of lists of tensors, one for each image in the batch. The outer list is over batch entries,
        the inner list is over subsequence entries. The tensors are of shape [c, h, w].
        Critical: only images that are actually present (as indicated by the image_present tensor) are included.
        All images in this list are expected to go through the image encoder and used for model input.
    image_input_ids: A list of lists of tensors. Outer list is over batch entries, inner list is over subsequence.
        These are the ids used as input to the model. For the ids that are `image_placeholder_id`, the corresponding
        encoded image patch embedding will be swapped in for model input. For variable-sized images, every raster
        line of patches is terminated with `image_newline_id`.
        Critical: there will be a tensor for every subsequence, even ones without images. Subsequences without
        images will just have a 0-length tensor. The number of placeholder ids in these lists must match the
        number of encoded image patches.
    image_patches: A list of lists of tensors, corresponding to `images` but with the images "patchified".
    image_patch_indices_per_batch: A list of lists of tensors. Outer list is over batch entries, inner list is over
        subsequence. These are the indices of the image patches in the token stream. Non-negative values correspond
        to image patches to be inserted in the stream.
    image_patch_indices_per_subsequence: Same as image_patch_indices_per_batch, but the indices are relative to
        the subsequence rather than the batch (i.e., the index counter is reset at each subsequence boundary).
        This is useful for reconstructing an individual image from predicted patches regardless of which subsequence
        it was in.
    """

    images: List[List[torch.Tensor]]
    image_input_ids: List[List[torch.Tensor]]
    image_patches: List[List[torch.Tensor]]
    image_patch_indices_per_batch: List[List[torch.Tensor]] = []
    image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []

    def to_device(self, device: torch.device) -> "ModelImageInput":
        def apply_nested(nested_list: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
            return [[x.to(device) for x in l] for l in nested_list]

        return ModelImageInput(
            images=apply_nested(self.images),
            image_input_ids=apply_nested(self.image_input_ids),
            image_patches=apply_nested(self.image_patches),
            image_patch_indices_per_batch=apply_nested(self.image_patch_indices_per_batch),
            image_patch_indices_per_subsequence=apply_nested(self.image_patch_indices_per_subsequence),
        )


def process_images_for_model_input(
    *,
    image_input: torch.Tensor,
    image_present: torch.Tensor,
    image_unpadded_h: torch.Tensor,
    image_unpadded_w: torch.Tensor,
    image_patch_dim_h: int,
    image_patch_dim_w: int,
    image_placeholder_id: int,
    image_newline_id: int,
    variable_sized: bool,
) -> ModelImageInput:
    """Process images for model input. In particular, variable-sized images are handled here.

    Args:
        image_input: [batch_size, num_sub_sequences, c, h, w] tensor of images.
        image_present: [batch_size, num_sub_sequences] tensor of 1s and 0s indicating whether an image is present.
        image_unpadded_h: [batch_size, num_sub_sequences] tensor of unpadded image heights.
        image_unpadded_w: [batch_size, num_sub_sequences] tensor of unpadded image widths.
        image_patch_dim_h: The height of the image patches.
        image_patch_dim_w: The width of the image patches.
        image_placeholder_id: The id of the image placeholder token.
        image_newline_id: The id of the image newline token.
        variable_sized: Whether to process images as variable-sized.
    """
    # Only images that are present.
    images: List[List[torch.Tensor]] = []
    image_patches: List[List[torch.Tensor]] = []
    # Image input ids for every subsequence, including ones with no image present.
    image_input_ids: List[List[torch.Tensor]] = []
    for bi in range(image_input.shape[0]):
        images.append([])
        image_input_ids.append([])
        image_patches.append([])
        for si in range(image_input.shape[1]):
            if image_present[bi, si]:
                image = image_input[bi, si]
                if variable_sized:
                    # The min() is required here due to floating point issues:
                    # math.ceil(torch.tensor(300).cuda() / 30) == 11
                    new_h = min(
                        image.shape[1], math.ceil(image_unpadded_h[bi, si] / image_patch_dim_h) * image_patch_dim_h
                    )
                    new_w = min(
                        image.shape[2], math.ceil(image_unpadded_w[bi, si] / image_patch_dim_w) * image_patch_dim_w
                    )
                    image = image[:, :new_h, :new_w]
                images[bi].append(image)
                num_patches = get_num_patches(
                    img_h=image.shape[1],
                    img_w=image.shape[2],
                    patch_dim_h=image_patch_dim_h,
                    patch_dim_w=image_patch_dim_w,
                )
                ids = torch.full([num_patches], image_placeholder_id, dtype=torch.int32, device=image_input.device)
                print(ids.shape)
                patches = patchify_image(
                    image=image.unsqueeze(0), patch_dim_h=image_patch_dim_h, patch_dim_w=image_patch_dim_w
                ).squeeze(0)
                if variable_sized:
                    # Now terminate each line with |NEWLINE|.
                    ids = ids.reshape(-1, new_w // image_patch_dim_w)
                    ids = torch.cat(
                        [
                            ids,
                            torch.full(
                                [ids.shape[0], 1], image_newline_id, dtype=torch.int32, device=image_input.device
                            ),
                        ],
                        dim=1,
                    )
                    ids = ids.reshape(-1)
                image_input_ids[bi].append(ids)
                image_patches[bi].append(patches)
            else:
                image_input_ids[bi].append(torch.tensor([], dtype=torch.int32, device=image_input.device))

    # Create image_patch_input_indices, where non-negative values correspond to image patches to be inserted in
    # the stream.
    image_patch_indices_per_batch: List[List[torch.Tensor]] = []
    image_patch_indices_per_subsequence: List[List[torch.Tensor]] = []
    for bi in range(len(image_input_ids)):
        image_patch_indices_per_batch.append([])
        image_patch_indices_per_subsequence.append([])
        index_offset = 0
        for si in range(len(image_input_ids[bi])):
            # Indices of image patches.
            num_patches = torch.count_nonzero(image_input_ids[bi][si] == image_placeholder_id)
            indices = torch.arange(
                num_patches,
                dtype=image_input_ids[bi][si].dtype,
                device=image_input_ids[bi][si].device,
            )

            # Place those indices in the image input ids token stream, with -1 representing non-index tokens.
            indices_in_stream_per_batch = torch.full_like(image_input_ids[bi][si], -1)
            indices_in_stream_per_subsequence = torch.full_like(image_input_ids[bi][si], -1)
            indices_in_stream_per_batch[
                torch.nonzero(image_input_ids[bi][si] == image_placeholder_id, as_tuple=True)[0]
            ] = (indices + index_offset)
            indices_in_stream_per_subsequence[
                torch.nonzero(image_input_ids[bi][si] == image_placeholder_id, as_tuple=True)[0]
            ] = indices

            image_patch_indices_per_batch[bi].append(indices_in_stream_per_batch)
            image_patch_indices_per_subsequence[bi].append(indices_in_stream_per_subsequence)
            index_offset += num_patches

    return ModelImageInput(
        images=images,
        image_input_ids=image_input_ids,
        image_patches=image_patches,
        image_patch_indices_per_batch=image_patch_indices_per_batch,
        image_patch_indices_per_subsequence=image_patch_indices_per_subsequence,
    )


class TextSegmentForTokenConversion(NamedTuple):
    """
    This named tuple saves a prompt as follows:
    - text: The text of the prompt segment
    - is_location: Whether the prompt segment is a bounding box or not

    Eg: this is a prompt <point>1, 2, 3, 4</point>
    will be saved as:
    [
        TextSegmentForTokenConversion(text="this is a prompt ", is_location=False),
        TextSegmentForTokenConversion(text="<point>1, 2, 3, 4</point>", is_location=True),
    ]
    """

    text: str
    is_location: bool


def _transform_coordinates_and_tokenize(
    prompt: str, transformed_image, tokenizer
) -> List[int]:
    """
    This function transforms the prompt in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompt tokens with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format:
    <box>y1, x1, y2, x2</box>
    <point>x, y</point>
    The spaces and punctuation added above are NOT optional.
    """
    # Make a namedtuple that stores "text" and "is_bbox"

    # We want to do the following: Tokenize the code normally -> when we see a point or box, tokenize using the tokenize_within_tag function
    # When point or box close tag, continue tokenizing normally
    # First, we replace the point and box tags with their respective tokens
    prompt = _replace_string_repr_with_token_tags(prompt)
    # Tokenize the prompt
    # Convert prompt into a list split
    prompt_text_list: List[TextSegmentForTokenConversion] = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem.is_location:
            # This is a location, we need to tokenize it
            within_tag_tokenized = _transform_within_tags(elem.text, transformed_image, tokenizer)
            # Surround the text with the open and close tags
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(tokenizer(elem.text, add_special_tokens=False).input_ids)
    return transformed_prompt_tokens


def _replace_string_repr_with_token_tags(prompt: str) -> str:
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> List:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # Wherever, we notice the [TOKEN_OPEN_STRING, TOKEN_CLOSE_STRING], we split the prompt
    prompt_text_list: List = []
    regex_pattern = re.compile(
        f"({TOKEN_BBOX_OPEN_STRING}|{TOKEN_BBOX_CLOSE_STRING}|{TOKEN_POINT_OPEN_STRING}|{TOKEN_POINT_CLOSE_STRING})"
    )
    # Split by the regex pattern
    prompt_split = regex_pattern.split(prompt)
    for i, elem in enumerate(prompt_split):
        if len(elem) == 0 or elem in [
            TOKEN_BBOX_OPEN_STRING,
            TOKEN_BBOX_CLOSE_STRING,
            TOKEN_POINT_OPEN_STRING,
            TOKEN_POINT_CLOSE_STRING,
        ]:
            continue
        prompt_text_list.append(
            TextSegmentForTokenConversion(
                text=elem,
                is_location=i > 1 and prompt_split[i - 1] in [TOKEN_BBOX_OPEN_STRING, TOKEN_POINT_OPEN_STRING],
            )
        )
    return prompt_text_list


def _transform_within_tags(text: str, transformed_image, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point>
    This function is responsible for converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """
    # Convert the text into a list of strings.
    num_int_strs = text.split(",")
    if len(num_int_strs) == 2:
        # If there are any open or close tags, remove them.
        token_space_open_string = tokenizer.vocab[TOKEN_POINT_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_POINT_CLOSE_STRING]
    else:
        token_space_open_string = tokenizer.vocab[TOKEN_BBOX_OPEN_STRING]
        token_space_close_string = tokenizer.vocab[TOKEN_BBOX_CLOSE_STRING]

    # Remove all spaces from num_ints
    num_ints = [float(num.strip()) for num in num_int_strs]
    # scale to transformed image siz
    if len(num_ints) == 2:
        num_ints_translated = scale_point_to_transformed_image(
            x=num_ints[0], y=num_ints[1], transformed_image=transformed_image
        )
    elif len(num_ints) == 4:
        num_ints_translated = scale_bbox_to_transformed_image(
            top=num_ints[0],
            left=num_ints[1],
            bottom=num_ints[2],
            right=num_ints[3],
            transformed_image=transformed_image,
        )
    else:
        raise ValueError(f"Invalid number of ints: {len(num_ints)}")
    # Tokenize the text, skipping the
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    transformed_images: Optional[List[List[torch.Tensor]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts
      plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them
      into a 3D tensor.
    """
    # args = get_args()
    # Tokenize all the prompts.
    # tokenizer = get_tokenizer()

    # If not tool use, tranform the coordinates while tokenizing
    if transformed_images is not None:
        assert len(prompts) == len(transformed_images)
        transformed_prompt_tokens = []
        for prompt_seq, transformed_image_seq in zip(prompts, transformed_images):
            transformed_prompt_tokens.append(
                [
                    _transform_coordinates_and_tokenize(prompt, transformed_image, tokenizer)
                    for prompt, transformed_image in zip(prompt_seq, transformed_image_seq)
                ]
            )
    else:
        transformed_prompt_tokens = [[tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts]

    prompts_tokens = transformed_prompt_tokens

    if add_BOS:
        bos_token = tokenizer.vocab['<s>']
    else:
        bos_token = tokenizer.vocab['|ENDOFTEXT|']
        prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        # Only add bbox open token to the last subsequence since that is what will be completed
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)

    # Now we have a list of list of tokens which each list has a different
    # size. We want to extend this list to:
    #   - incorporate the tokens that need to be generated
    #   - make all the sequences equal length.
    # Get the prompts length.

    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]
    # Get the max prompts length.
    max_prompt_len: int = np.max(prompts_length)
    # Number of tokens in the each sample of the batch.
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        print(
            f"Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}",
            f"exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.",
        )
    # Now update the list of list to be of the same size: samples_length.
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError("Length of subsequence prompt exceeds sequence length.")
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab['|ENDOFTEXT|']] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)  # , device="cuda")
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)  # , device="cuda")

    return prompts_tokens_tensor, prompts_length_tensor


def tokenize_prompts_with_images(
    tokenizer,
    prompts: Optional[List[List[str]]],
    transformed_images: Optional[List[List[TransformedImage]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,
    add_beginning_of_answer_token: bool,
    rank: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompts and make them avaiable on all ranks."""
    assert add_BOS is not None

    # On all ranks set to None so we can pass them to functions
    sizes_list = None
    prompts_tokens_cuda_long_tensor = None
    prompts_length_cuda_long_tensor = None

    assert prompts is not None
    assert max_tokens_to_generate is not None
    # Tensor of tokens padded and their unpadded length.
    prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = _tokenize_prompts_with_image_and_batch(
        tokenizer,
        prompts,
        transformed_images,
        max_tokens_to_generate,
        max_position_embeddings,
        add_BOS,
        add_beginning_of_answer_token,
    )
    # We need the sizes of these tensors for the broadcast
    sizes_list = [
        prompts_tokens_cuda_long_tensor.size(0),  # Batch size
        prompts_tokens_cuda_long_tensor.size(1),  # Num subsequences
        prompts_tokens_cuda_long_tensor.size(2),  # Sequence length
    ]
    # First, broadcast the sizes.
    sizes_tensor = broadcast_int_list(3, int_list=sizes_list, rank=rank)

    # Now that we have the sizes, we can broadcast the tokens
    # and length tensors.
    sizes = sizes_tensor.tolist()
    """prompts_tokens_cuda_long_tensor = broadcast_tensor(
        sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank
    )
    prompts_length_cuda_long_tensor = broadcast_tensor(
        sizes[:2], torch.int64, tensor=prompts_length_cuda_long_tensor, rank=rank
    )"""

    return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()


def broadcast_tensor(size, dtype, tensor=None, rank=0, device=0):
    """Given size and type of a tensor on all ranks and the tensor value
    only on a specific rank, broadcast from that rank to all other ranks.
    Args:
        size: size of the tensor
        dtype: type of the tensor
        tensor: tensor to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    if device is not None:
        device = torch.cuda.current_device()
    if device is not None:
        _is_cuda_contiguous(tensor)
    else:
        tensor = torch.empty(size, dtype=dtype, device=device)
    return tensor


def broadcast_list(size, dtype, list_values=None, rank=0, device=0):
    """Broadcast a list of values with a given type.
    Args:
        size: size of the list
        dtype: dtype of the list
        list_values: list of values to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    tensor = None
    if device is not None:
        device = torch.cuda.current_device()

    tensor = torch.tensor(list_values, dtype=dtype, device=device)
    return broadcast_tensor(size, dtype, tensor=tensor, rank=rank, device=device)


def broadcast_int_list(size, int_list=None, rank=0, device=0):
    """Broadcast a list of interger values.
    Args:
        size: size of the list
        int_list: list of values to be broadcasted
        rank: primary rank for broadcasting
        device: device of the tensor. If not set to None, then we use cuda.current_device().
            Default is 0, since we use cuda.current_device() to get the device.
    """
    if device is not None:
        device = torch.cuda.current_device()
    return broadcast_list(size, torch.int64, list_values=int_list, rank=rank, device=device)


def ensure_tensor(input_image: Optional[Union[Image.Image, npt.NDArray, torch.Tensor]]) -> Optional[torch.Tensor]:
    if isinstance(input_image, Image.Image):
        return TF.to_tensor(input_image)
    if isinstance(input_image, np.ndarray):
        return torch.tensor(input_image)
    return input_image


# TODO(fjord): Once we're using the new vocabulary, switch to reserved/prioritized tokens like <bbox> and <point>.
TOKEN_BBOX_OPEN_STRING = BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_BBOX_CLOSE_STRING = TOKEN_POINT_OPEN_STRING = POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>

T = TypeVar("T")


class TransformedImage(NamedTuple):
    """Keep track of padding and resizing operations that resulted in a given image.

    Canonical order of transformation is crop -> scale -> pad

    Attributes:
        image: Image to feed into the model for training or inference. Shape: [channels, height, width]
        crop_top:     Top coord of visible rect in orig coord space.
        crop_left:    Left coord of      ".
        crop_bottom:  Bottom coord of    ".
        crop_right:   Right coord of     ".
        scaled_h:     Height of the whole image (without cropping or padding) after resizing.
        scaled_w:     Width of the whole image (without cropping or padding) after resizing.
        padding_top:     Size of padding above    the image
        padding_left:                    left of
        transformed_h: Height of the image after all transforms
        transformed_w: Width of the image after all transforms
        original_h: Height of the image before any transforms.
        original_w: Width of the image before any transforms.
        rng: Generator to use for any random modifications.
    """

    image: torch.Tensor
    crop_top: int
    crop_left: int
    crop_bottom: int
    crop_right: int
    scaled_h: int
    scaled_w: int
    padding_top: int
    padding_left: int
    transformed_h: int
    transformed_w: int
    original_h: int
    original_w: int
    rng: np.random.Generator

    @classmethod
    def from_image(
        cls, image: Union[Image.Image, npt.NDArray, torch.Tensor], rng_seed: Optional[int] = None
    ) -> "TransformedImage":
        """Simple constructor to create a TransformedImage without transformations"""
        # three-dimensional tensor with shape = (# channels, height, width)
        image_tensor = ensure_tensor(image)
        _, height, width = TF.get_dimensions(image_tensor)

        return TransformedImage(
            image=image_tensor,
            crop_top=0,
            crop_left=0,
            crop_bottom=height,
            crop_right=width,
            scaled_h=height,
            scaled_w=width,
            padding_top=0,
            padding_left=0,
            transformed_h=height,
            transformed_w=width,
            original_h=height,
            original_w=width,
            rng=np.random.default_rng(rng_seed),
        )

    @property
    def unpadded_h(self) -> int:
        """Returns the height of the image after crop + resize"""
        crop_h = self.crop_bottom - self.crop_top
        factor = self.scaled_h / self.original_h
        return int(crop_h * factor)

    @property
    def unpadded_w(self) -> int:
        """Returns the width of the image after crop + resize"""
        crop_w = self.crop_right - self.crop_left
        factor = self.scaled_w / self.original_w
        return int(crop_w * factor)

    def _scale_coords(
        self, input_coords: Union[npt.NDArray, torch.Tensor], scale: float
    ) -> Union[npt.NDArray, torch.Tensor]:
        if isinstance(input_coords, torch.Tensor):
            return torch.round(input_coords * scale).to(torch.int32)
        elif isinstance(input_coords, np.ndarray):
            return np.round(input_coords * scale).astype(np.int32)
        else:
            raise ValueError(f"Unknown coordinate type: {input_coords}.")

    def _clamp_coords(
        self, input_coords: Union[npt.NDArray, torch.Tensor], min_value: int, max_value: int
    ) -> Union[npt.NDArray, torch.Tensor]:
        if isinstance(input_coords, torch.Tensor):
            return torch.clamp(input_coords, min_value, max_value - 1)
        elif isinstance(input_coords, np.ndarray):
            return np.clip(input_coords, min_value, max_value - 1)
        else:
            raise ValueError(f"Unknown coordinate type: {input_coords}.")

    def original_to_transformed_h_coords(
        self, original_coords: Union[npt.NDArray, torch.Tensor]
    ) -> Union[npt.NDArray, torch.Tensor]:
        # apply crop
        cropped_coords = (
            self._clamp_coords(original_coords, min_value=self.crop_top, max_value=self.crop_bottom) - self.crop_top
        )
        # apply scale
        scaled_coords = self._scale_coords(cropped_coords, scale=self.scaled_h / self.original_h)
        # apply pad
        return scaled_coords + self.padding_top

    def original_to_transformed_w_coords(
        self, original_coords: Union[npt.NDArray, torch.Tensor]
    ) -> Union[npt.NDArray, torch.Tensor]:
        # apply crop
        cropped_coords = (
            self._clamp_coords(original_coords, min_value=self.crop_left, max_value=self.crop_right) - self.crop_left
        )
        # apply scale
        scaled_coords = self._scale_coords(cropped_coords, scale=self.scaled_w / self.original_w)
        # apply pad
        return scaled_coords + self.padding_left

    def transformed_to_original_h_coords(
        self, transformed_coords: Union[npt.NDArray, torch.Tensor]
    ) -> Union[npt.NDArray, torch.Tensor]:
        # apply inv pad
        unpadded_coords = self._clamp_coords(
            transformed_coords - self.padding_top, min_value=0, max_value=self.unpadded_h
        )
        # apply inv scale
        unscaled_coords = self._scale_coords(unpadded_coords, scale=self.original_h / self.scaled_h)
        # apply inv crop
        return unscaled_coords + self.crop_top

    def transformed_to_original_w_coords(
        self, transformed_coords: Union[npt.NDArray, torch.Tensor]
    ) -> Union[npt.NDArray, torch.Tensor]:
        # apply inv pad
        unpadded_coords = self._clamp_coords(
            transformed_coords - self.padding_left, min_value=0, max_value=self.unpadded_w
        )
        # apply inv scale
        unscaled_coords = self._scale_coords(unpadded_coords, scale=self.original_w / self.scaled_w)
        # apply inv crop
        return unscaled_coords + self.crop_left


def scale_point_to_transformed_image(x: float, y: float, transformed_image: TransformedImage) -> List[int]:
    x_scaled = transformed_image.original_to_transformed_w_coords(np.array([x / 2]))[0]
    y_scaled = transformed_image.original_to_transformed_h_coords(np.array([y / 2]))[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, transformed_image: TransformedImage
) -> List[int]:
    top_scaled = transformed_image.original_to_transformed_w_coords(np.array([top / 2]))[0]
    left_scaled = transformed_image.original_to_transformed_h_coords(np.array([left / 2]))[0]
    bottom_scaled = transformed_image.original_to_transformed_w_coords(np.array([bottom / 2]))[0]
    right_scaled = transformed_image.original_to_transformed_h_coords(np.array([right / 2]))[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


def permute_list(rng: np.random.Generator, inp: Sequence[T]) -> List[T]:
    """Use rng.permutation to randomly permute a list."""
    # This is better than converting the list to an np.array and having permutation do it directly
    # because going to/from np.array ends up messing with our tuple types (even if you specify dtype=object).
    answer: List[T] = []
    permutation = rng.permutation(len(inp)).tolist()
    for i in permutation:
        answer.append(inp[i])
    return answer
