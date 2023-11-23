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
Image/Text processor class for GIT
"""
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends


if is_torch_available():
    from .image_processing_tomato import TomatoBatchFeature


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>

def mark_continuous_neg_ones(lst):
    """
    Returns a list of boolean values where True indicates the position of at least two continuous -1s in the input list.
    
    :param lst: List of integers.
    :return: List of boolean values.
    """
    result = [False] * len(lst)  # Initialize the result list with False
    i = 0  # Start index

    while i < len(lst):
        if lst[i] == -1:
            # Check if it's a start of a continuous sequence of -1
            start = i
            while i < len(lst) and lst[i] == -1:
                i += 1
            end = i
            # Mark all positions in the sequence as True only if the sequence length is at least 2
            if end - start > 1:
                for j in range(start, end):
                    result[j] = True
        else:
            i += 1

    return result

def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: List[int],
    full_unpacked_stream: List["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
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
        new_padded_tensor[bi, :tokens_to_place] = full_unpacked_stream[bi][offset : tokens_to_place + offset]

    return new_padded_tensor


def construct_full_unpacked_stream(
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    image_patch_indices_stream: "torch.Tensor",
    image_patch_indices: List[List["torch.Tensor"]],
    image_indicator_id: int,
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    all_bi_stream, all_bi_image_patch_indices_stream = [], []

    for batch_index in range(batch_size):
        # Extract the subsequence from the input_stream; assume only one subsequence in text
        subsequence_stream = input_stream[batch_index, 0].clone()
        subsequence_patch_indices_stream = image_patch_indices_stream[batch_index, 0].clone()
        

        # If there are image tokens for this batch item
        if image_tokens[batch_index]:
            # Find indices where image_indicator_id appears
            image_indicator_indices = (subsequence_stream == image_indicator_id).nonzero(as_tuple=True)[0]

            # Assert that the number of image indicators matches the number of image tokens
            assert len(image_indicator_indices) == len(image_tokens[batch_index]), \
                "Number of image indicators does not match the number of image tokens."

            # Replace image_indicator_id with actual image tokens
            offset = 0
            for idx, image_token, image_patch_indice in zip(image_indicator_indices, image_tokens[batch_index], image_patch_indices[batch_index]):
                adjusted_idx = idx + offset
                subsequence_stream = torch.cat([subsequence_stream[:adjusted_idx], image_token, subsequence_stream[adjusted_idx+1:]])
                subsequence_patch_indices_stream = torch.cat([subsequence_patch_indices_stream[:adjusted_idx], image_patch_indice, subsequence_patch_indices_stream[adjusted_idx+1:]])
                offset += len(image_token) - 1  # Adjust offset for subsequent replacements

        all_bi_stream.append(subsequence_stream)
        all_bi_image_patch_indices_stream.append(subsequence_patch_indices_stream)

    return all_bi_stream, all_bi_image_patch_indices_stream


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
            (elem, i > 1 and prompt_split[i - 1] in [TOKEN_BBOX_OPEN_STRING, TOKEN_POINT_OPEN_STRING])
        )
    return prompt_text_list


def _transform_coordinates_and_tokenize(prompt: str, scale_factor: float, tokenizer) -> List[int]:
    """
    This function transforms the prompt in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompt tokens with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
    and punctuation added above are NOT optional.
    """
    # Make a namedtuple that stores "text" and "is_bbox"

    # We want to do the following: Tokenize the code normally -> when we see a point or box, tokenize using the tokenize_within_tag function
    # When point or box close tag, continue tokenizing normally
    # First, we replace the point and box tags with their respective tokens
    prompt = _replace_string_repr_with_token_tags(prompt)
    # Tokenize the prompt
    # Convert prompt into a list split
    prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
    transformed_prompt_tokens: List[int] = []
    for elem in prompt_text_list:
        if elem[1]:
            # This is a location, we need to tokenize it
            within_tag_tokenized = _transform_within_tags(elem[0], scale_factor, tokenizer)
            # Surround the text with the open and close tags
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(tokenizer(elem[0], add_special_tokens=False).input_ids)
    return transformed_prompt_tokens


def _transform_within_tags(text: str, scale_factor: float, tokenizer) -> List[int]:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
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
        num_ints_translated = scale_point_to_transformed_image(x=num_ints[0], y=num_ints[1], scale_factor=scale_factor)
    elif len(num_ints) == 4:
        num_ints_translated = scale_bbox_to_transformed_image(
            top=num_ints[0],
            left=num_ints[1],
            bottom=num_ints[2],
            right=num_ints[3],
            scale_factor=scale_factor,
        )
    else:
        raise ValueError(f"Invalid number of ints: {len(num_ints)}")
    # Tokenize the text, skipping the
    tokens = [tokenizer.vocab[str(num)] for num in num_ints_translated]
    return [token_space_open_string] + tokens + [token_space_close_string]


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: List[List[str]],
    scale_factors: Optional[List[List["torch.Tensor"]]],
    image_tokens: Optional[List["torch.Tensor"]],
    image_indicator_id: int,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    if image_tokens is not None:
        image_indicator_prompt = tokenizer.decode(image_indicator_id)
        for i, single_prompt in enumerate(prompts):
            for j, (prompt, image_token) in enumerate(zip(single_prompt, image_tokens)):
                image_indicator_count = prompt.count(image_indicator_prompt)
                if image_indicator_count > len(image_token):
                    raise ValueError(f"Image place indicators exceed the number of images provided. Have {image_indicator_count} images?")
                elif image_indicator_count < len(image_token):
                    insert_count = len(image_token) - image_indicator_count
                    logger.warning(f"Inserting {insert_count} image place indicators before the prompt.")
                    prompt = image_indicator_prompt * insert_count + prompt

                assert prompt.count(image_indicator_prompt) == len(image_token)
                # Write back the modified prompt to the prompts list
                prompts[i][j] = prompt


    # If not tool use, tranform the coordinates while tokenizing
    if scale_factors is not None:
        transformed_prompt_tokens = []
        for prompt_seq, scale_factor_seq in zip(prompts, scale_factors):
            transformed_prompt_tokens.append(
                [
                    _transform_coordinates_and_tokenize(prompt, scale_factor.item(), tokenizer)
                    for prompt, scale_factor in zip(prompt_seq, scale_factor_seq)
                ]
            )
    else:
        transformed_prompt_tokens = [[tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts]

    prompts_tokens = transformed_prompt_tokens

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)

    return prompts_tokens_tensor


# Simplified assuming self.crop_top = self.padding_top = 0
def original_to_transformed_h_coords(original_coords, scale_h):
    return np.round(original_coords * scale_h).astype(np.int32)


# Simplified assuming self.crop_left = self.padding_left = 0
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)


def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> List[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]), scale_factor)[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]), scale_factor)[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, scale_factor: float
) -> List[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]), scale_factor)[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]), scale_factor)[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]), scale_factor)[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]), scale_factor)[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


class TomatoProcessor(ProcessorMixin):
    r"""
    Constructs a Tomato processor which wraps a Tomato image processor and a Llama tokenizer into a single processor.

    [`TomatoProcessor`] offers all the functionalities of [`TomatoImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~TomatoProcessor.__call__`] and [`~TomatoProcessor.decode`] for more information.

    Args:
        image_processor ([`TomatoImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "TomatoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 0 # don't know why this is important
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1

    def _left_pad_inputs_with_attention_mask(self, model_inputs: List[Dict], return_attention_mask: bool, truncation: bool, truncation_length: int):
        max_length_input_ids = min(max(entry["input_ids"].shape[1] for entry in model_inputs), truncation_length)
        max_length_image_patch_indices = min(max(entry["image_patches_indices"].shape[1] for entry in model_inputs), truncation_length)

        batched_inputs = {"input_ids": [], "image_patches": [], "image_patches_indices": [], "attention_mask": []}

        for entry in model_inputs:
            for key, tensor in entry.items():
                if key == "input_ids" or key == "image_patches_indices":
                    # Truncate if the tensor is longer than the truncation_length
                    if tensor.shape[1] > truncation_length:
                        if truncation:
                            logger.warn(f"Truncating tensor from original length {tensor.shape[1]} to {truncation_length}")
                            tensor = tensor[:, :truncation_length]
                        else:
                            raise ValueError(f"Tensor length {tensor.shape[1]} exceeds truncation_length {truncation_length} (usually max positional embedding length), but truncation is disabled. Please enable truncation or increase truncation_length.")
                    
                    # Calculate the number of padding tokens or indices
                    num_padding = max_length_input_ids - tensor.shape[1] if key == "input_ids" else max_length_image_patch_indices - tensor.shape[1]
                    
                    # Pad the tensor
                    padded_tensor = torch.cat(
                        [
                            torch.full((tensor.shape[0], num_padding), self.pad_token_id if key == "input_ids" else self.dummy_image_index, dtype=torch.long),
                            tensor,
                        ],
                        dim=1,
                    )
                    batched_inputs[key].append(padded_tensor)

                    if key == "input_ids":
                        attention_mask = torch.cat(
                            [torch.zeros(tensor.shape[0], num_padding, dtype=torch.long), torch.ones_like(tensor)],
                            dim=1,
                        )
                        batched_inputs["attention_mask"].append(attention_mask)

                elif key == "image_patches":
                    # For image_patches, we don't pad but just append them to the list.
                    batched_inputs[key].append(tensor)

        batched_keys = ["input_ids", "image_patches_indices"]
        if return_attention_mask:
            batched_keys.append("attention_mask")
        for key in batched_keys:
            batched_inputs[key] = torch.cat(batched_inputs[key], dim=0)

        batched_inputs["labels"] = self.get_labels(batched_inputs["input_ids"], batched_inputs["image_patches_indices"])

        return batched_inputs

    def get_sample_encoding(
        self,
        prompts,
        scale_factors,
        image_unpadded_heights,
        image_unpadded_widths,
        image_indicator_id,
        image_placeholder_id,
        image_newline_id,
        tensor_batch_images,
    ):
        image_present = torch.ones(tensor_batch_images.shape[0], tensor_batch_images.shape[1], 1) #  shape [batch_size, subsequence_size, num_images]
        model_image_input = self.image_processor.preprocess_with_tokenizer_info(
            image_input=tensor_batch_images,
            image_present=image_present,
            image_unpadded_h=image_unpadded_heights,
            image_unpadded_w=image_unpadded_widths,
            image_placeholder_id=image_placeholder_id,
            image_newline_id=image_newline_id,
            variable_sized=True,
        )
        # FIXME max_tokens_to_generate is embedded into this processor's call.
        prompt_tokens = _tokenize_prompts_with_image_and_batch(
            tokenizer=self.tokenizer,
            prompts=prompts,
            scale_factors=scale_factors,
            image_tokens=model_image_input["image_input_ids"],
            image_indicator_id=image_indicator_id,
        )
        image_padded_unpacked_tokens, unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
            input_stream=prompt_tokens,
            image_tokens=model_image_input["image_input_ids"],
            image_patch_indices_stream=torch.full_like(prompt_tokens, -1),
            image_patch_indices=model_image_input["image_patch_indices_per_batch"],
            image_indicator_id=image_indicator_id,
            batch_size=1,
            num_sub_sequences=self.subsequence_length,
        )
        max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
        max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
        tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[0].shape[0]))

        # Use same packing logic for the image patch indices.
        image_patch_input_indices = full_unpacked_stream_to_tensor(
            all_bi_tokens_to_place=[tokens_to_place],
            full_unpacked_stream=unpacked_image_patch_indices_per_batch,
            fill_value=-1,
            batch_size=1,
            new_seq_len=max_seq_len_batch,
            offset=0,
        )
        image_patches_tensor = torch.stack([torch.concat(img, dim=0) for img in model_image_input["image_patches"]])
        batch_encoding = {
            "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
            "image_patches": image_patches_tensor,
            "image_patches_indices": image_patch_input_indices,
        }
        return batch_encoding

    def __call__(
        self,
        text=None,
        images=None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> "TomatoBatchFeature":
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        TomatoImageProcessor's [`~TomatoImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:
            [`TomatoBatchEncoding`]: A [`TomatoBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ["torch"])

        # --- Check input validity ---
        if not return_attention_mask:
            raise ValueError("`return_attention_mask=False` is not supported for this model.")
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be None.")
        if text is not None and images is None:
            logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            max_length = self.max_position_embeddings if max_length is None else max_length
            self.current_processor = self.tokenizer
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
            return text_encoding

        if text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            prompts = [[""]]
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq if text_seq is not None else ""] for text_seq in text]

        # --- Preprocess images using self.image_processor ---

        # FIXME - We hard code "pt" here because the rest of the processing assumes torch tensors
        image_encoding = self.image_processor.preprocess(images, return_tensors="pt")
        batch_images = image_encoding["images"]
        image_unpadded_heights = image_encoding["image_unpadded_heights"]
        image_unpadded_widths = image_encoding["image_unpadded_widths"]
        scale_factors = image_encoding["image_scale_factors"]
        self.subsequence_length = 1  # Each batch contains only one sequence.
        self.batch_size = len(batch_images)

        # --- Use self.tokenizer to get the ids of special tokens to insert into image ids ---

        image_indicator_id = self.tokenizer("<|Image|>", add_special_tokens=False)["input_ids"][0]
        image_placeholder_id = self.tokenizer("<|SPEAKER|>", add_special_tokens=False)["input_ids"][0] # add both tokens to tokenizer.json
        image_newline_id = self.tokenizer("<|NEWLINE|>", add_special_tokens=False)["input_ids"][0]
        tensor_batch_images = [torch.stack(batch_image).unsqueeze(0) for batch_image in batch_images]

        # --- Use self.image_processor again to obtain the full token ids and batch inputs ---
        all_encodings = []

        for prompt, scale_factor, image_unpadded_height, image_unpadded_width, tensor_batch_image in zip(
            prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, tensor_batch_images
        ):
            sample_encoding = self.get_sample_encoding(
                prompts=[prompt],
                scale_factors=[scale_factor],
                image_unpadded_heights=torch.tensor([image_unpadded_height]),
                image_unpadded_widths=torch.tensor([image_unpadded_width]),
                image_indicator_id=image_indicator_id,
                image_placeholder_id=image_placeholder_id,
                image_newline_id=image_newline_id,
                tensor_batch_images=tensor_batch_image,
            )
            all_encodings.append(sample_encoding)
        batch_encoding = self._left_pad_inputs_with_attention_mask(
            model_inputs=all_encodings, return_attention_mask=return_attention_mask, truncation=truncation, truncation_length=max_length if max_length is not None else self.max_position_embeddings,
        )
        return TomatoBatchFeature(data=batch_encoding)
    
    def get_labels(self, input_ids, image_patches_indices, special_token_id=-1, masking_number=-100):
        """
        Mask the labels of image part.
        """
        labels = torch.full_like(input_ids, masking_number)
        for i in range(input_ids.shape[0]):
            seq = image_patches_indices[i]
            # indices = mark_continuous_neg_ones(seq)[:input_ids.shape[1]]
            indices = (seq != special_token_id).nonzero(as_tuple=True)[0]
            labels[i, indices] = input_ids[i, indices]
        return labels
            

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        """
        Transforms raw coordinates detected by [`TomatoForCausalLM`] to the original images' coordinate space.
        Coordinates will be returned in "box" format, with the following pattern:
            `<box>top, left, bottom, right</box>`

        Point coordinates are not supported yet.

        Args:
            outputs ([`GenerateOutput`]):
                Raw outputs from `generate`.
            target_sizes (`torch.Tensor`, *optional*):
                Tensor of shape (batch_size, 2) where each entry is the (height, width) of the corresponding image in
                the batch. If set, found coordinates in the output sequence are rescaled to the target sizes. If left
                to None, coordinates will not be rescaled.

        Returns:
            `GenerateOutput`: Same output type returned by `generate`, with output token ids replaced with
                boxed and possible rescaled coordinates.
        """

        def scale_factor_to_fit(original_size, target_size=None):
            height, width = original_size
            if target_size is None:
                max_height = self.image_processor.size["height"]
                max_width = self.image_processor.size["width"]
            else:
                max_height, max_width = target_size
            if width <= max_width and height <= max_height:
                return 1.0
            return min(max_height / height, max_width / width)

        def find_delimiters_pair(tokens, start_token, end_token):
            start_id = self.tokenizer.convert_tokens_to_ids(start_token)
            end_id = self.tokenizer.convert_tokens_to_ids(end_token)

            starting_positions = (tokens == start_id).nonzero(as_tuple=True)[0]
            ending_positions = (tokens == end_id).nonzero(as_tuple=True)[0]

            if torch.any(starting_positions) and torch.any(ending_positions):
                return (starting_positions[0], ending_positions[0])
            return (None, None)

        def tokens_to_boxes(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_BBOX_OPEN_STRING, TOKEN_BBOX_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 5:
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                top, left, bottom, right = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_BBOX_OPEN}{top}, {left}, {bottom}, {right}{TEXT_REPR_BBOX_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        def tokens_to_points(tokens, original_size):
            while (pair := find_delimiters_pair(tokens, TOKEN_POINT_OPEN_STRING, TOKEN_POINT_CLOSE_STRING)) != (
                None,
                None,
            ):
                start, end = pair
                if end != start + 3:
                    continue

                # Retrieve transformed coordinates from tokens
                coords = self.tokenizer.convert_ids_to_tokens(tokens[start + 1 : end])

                # Scale back to original image size and multiply by 2
                scale = scale_factor_to_fit(original_size)
                x, y = [2 * int(float(c) / scale) for c in coords]

                # Replace the IDs so they get detokenized right
                replacement = f" {TEXT_REPR_POINT_OPEN}{x}, {y}{TEXT_REPR_POINT_CLOSE}"
                replacement = self.tokenizer.tokenize(replacement)[1:]
                replacement = self.tokenizer.convert_tokens_to_ids(replacement)
                replacement = torch.tensor(replacement).to(tokens)

                tokens = torch.cat([tokens[:start], replacement, tokens[end + 1 :]], 0)
            return tokens

        if target_sizes is None:
            target_sizes = ((self.image_processor.size["height"], self.image_processor.size["width"]),) * len(outputs)
        elif target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        if len(outputs) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as output sequences")

        results = []
        for seq, size in zip(outputs, target_sizes):
            seq = tokens_to_boxes(seq, size)
            seq = tokens_to_points(seq, size)
            results.append(seq)

        return results

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
