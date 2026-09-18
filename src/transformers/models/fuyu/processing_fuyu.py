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

import math
import re
from typing import Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging, requires_backends
from ...utils.import_utils import requires


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


class FuyuProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
        },
    }


# Simplified assuming self.crop_top = self.padding_top = 0
def original_to_transformed_h_coords(original_coords, scale_h):
    return np.round(original_coords * scale_h).astype(np.int32)


# Simplified assuming self.crop_left = self.padding_left = 0
def original_to_transformed_w_coords(original_coords, scale_w):
    return np.round(original_coords * scale_w).astype(np.int32)


def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> list[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]), scale_factor)[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]), scale_factor)[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, scale_factor: float
) -> list[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]), scale_factor)[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]), scale_factor)[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]), scale_factor)[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]), scale_factor)[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


def _replace_string_repr_with_token_tags(prompt: str) -> str:
    prompt = prompt.replace(TEXT_REPR_POINT_OPEN, TOKEN_POINT_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_POINT_CLOSE, TOKEN_POINT_CLOSE_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_OPEN, TOKEN_BBOX_OPEN_STRING)
    prompt = prompt.replace(TEXT_REPR_BBOX_CLOSE, TOKEN_BBOX_CLOSE_STRING)
    return prompt


def _segment_prompt_into_text_token_conversions(prompt: str) -> list:
    """
    Given a string prompt, converts the prompt into a list of TextTokenConversions.
    """
    # Wherever, we notice the [TOKEN_OPEN_STRING, TOKEN_CLOSE_STRING], we split the prompt
    prompt_text_list: list = []
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


def transform_within_tags(text: str, scale_factor: float) -> str:
    """
    Given a bounding box of the fashion <box>1, 2, 3, 4</box> | <point>1, 2</point> This function is responsible for
    converting 1, 2, 3, 4 into tokens of 1 2 3 4 without any commas.
    """
    # Convert the text into a list of strings.
    num_int_strs = text.split(",")
    if len(num_int_strs) == 2:
        # If there are any open or close tags, remove them.
        token_space_open_string = TOKEN_POINT_OPEN_STRING
        token_space_close_string = TOKEN_POINT_CLOSE_STRING
    else:
        token_space_open_string = TOKEN_BBOX_OPEN_STRING
        token_space_close_string = TOKEN_BBOX_CLOSE_STRING

    # Remove all spaces from num_ints
    num_ints = [float(num.strip()) for num in num_int_strs]
    # scale to transformed image size
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
    tokens = "".join([str(num) for num in num_ints_translated])
    return token_space_open_string + tokens + token_space_close_string


def transform_coordinates(text: list[str], scale_factors: list["torch.Tensor"]) -> list[str]:
    """
    This function transforms the prompts in the following fashion:
    - <box> <point> and </box> </point> to their respective token mappings
    - extract the coordinates from the tag
    - transform the coordinates into the transformed image space
    - return the prompts with the transformed coordinates and new tags

    Bounding boxes and points MUST be in the following format: <box>y1, x1, y2, x2</box> <point>x, y</point> The spaces
    and punctuation added above are NOT optional.
    """
    processed_text = []
    for prompt, scale_factor in zip(text, scale_factors):
        prompt = _replace_string_repr_with_token_tags(prompt)
        prompt_text_list = _segment_prompt_into_text_token_conversions(prompt)
        transformed_prompts: list[int] = []
        for elem in prompt_text_list:
            if elem[1]:
                # This is a location, surround the text with the open and close tags
                prompt_within_tags = transform_within_tags(elem[0], scale_factor.item())
                transformed_prompts.append(prompt_within_tags)
            else:
                transformed_prompts.append(elem[0])
        processed_text.append("".join(transformed_prompts))
    return processed_text


def construct_full_unpacked_stream(
    num_real_text_tokens: Union[list[list[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: list[list["torch.Tensor"]],
    batch_size: int,
) -> list["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    subsequence_stream = torch.cat([image_tokens[0][0], input_stream[0, 0]], dim=0)
    num_real_tokens = image_tokens[0][0].shape[0] + num_real_text_tokens[0][0]
    all_bi_stream = [torch.cat([subsequence_stream[:num_real_tokens]], dim=0)]

    return all_bi_stream


@requires(backends=("vision",))
@auto_docstring
class FuyuProcessor(ProcessorMixin):
    valid_processor_kwargs = FuyuProcessorKwargs

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        """
        Override for BC. Fuyu uses TokenizersBackend and requires token_type_ids to be removed from model_input_names
        because Fuyu uses mm_token_type_ids instead for multimodal token identification.    `
        """
        from ...tokenization_utils_tokenizers import TokenizersBackend

        tokenizer = TokenizersBackend.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Remove token_type_ids as Fuyu uses mm_token_type_ids instead
        if "token_type_ids" in tokenizer.model_input_names:
            tokenizer.model_input_names.remove("token_type_ids")
        return tokenizer

    def __init__(self, image_processor, tokenizer, **kwargs):
        self.max_tokens_to_generate = 10
        vocab = tokenizer.get_vocab()
        tokenizer.pad_token_id = 0

        self.image_token = "|SPEAKER|"
        self.image_newline_token = "|NEWLINE|"
        self.image_token_id = vocab["|SPEAKER|"]
        self.image_newline_id = vocab["|NEWLINE|"]
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)

    @property
    def image_token_ids(self) -> list[int]:
        return [self.image_newline_id, self.image_token_id]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[FuyuProcessorKwargs],
    ) -> "BatchFeature":
        r"""
        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ["torch"])

        merged_kwargs = self._merge_kwargs(
            FuyuProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not merged_kwargs["text_kwargs"].setdefault("return_attention_mask", True):
            raise ValueError("`return_attention_mask=False` is not supported for this model.")

        # Was and still is hardcoded by manually padding/tensorizing all inputs due to BC
        if text is not None and images is not None:
            merged_kwargs["text_kwargs"]["add_special_tokens"] = False
            merged_kwargs["text_kwargs"]["padding"] = False

        if images is not None:
            merged_kwargs["text_kwargs"]["return_tensors"] = "pt"
            merged_kwargs["images_kwargs"]["return_tensors"] = "pt"

        # Processor has a ton of custom code to tokenize inputs and we need to keep it for BC
        # passing the whole seq to tokenizer adds SP underscore (`_`) randomly in the beginning
        images, text, *_ = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)

        processed_images = {}
        images_replacements = []
        if images is not None and hasattr(self, "image_processor"):
            processed_images, images_replacements = self._process_images(images, **merged_kwargs["images_kwargs"])

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            # if there is an image associated with text, process location tags
            # scale_factor are output from image processor, so we have to do it here
            if (image_scale_factors := processed_images.get("image_scale_factors")) is not None:
                sample = [transform_coordinates(sample, scale) for sample, scale in zip(text, image_scale_factors)]

            if images_replacements:
                text, text_replacement_offsets = self.get_text_with_replacements(
                    text,
                    images_replacements,
                )

            # IMPORTANT: here comes the custom part with tokenization, do not change it!
            if not images_replacements:
                text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            else:
                # encode the text and the placeholders separately, then always pad on the left
                batch_input_ids, batch_attention_mask = [], []
                for sample in text:
                    split_sample = re.split(r"(?<=<s>)", sample, maxsplit=1)
                    prompt_inputs = self.tokenizer(split_sample[-1], **merged_kwargs["text_kwargs"])

                    if len(split_sample) == 2:
                        # strip off the underscore which is always prepended before special image tokens
                        # we are guaranteed that the output are torch tensors
                        placeholder_inputs = self.tokenizer(split_sample[0], **merged_kwargs["text_kwargs"])
                        batch_input_ids.append(
                            torch.cat(
                                [placeholder_inputs["input_ids"][..., 1:], prompt_inputs["input_ids"]], dim=-1
                            ).squeeze(0)
                        )
                        batch_attention_mask.append(
                            torch.cat(
                                [placeholder_inputs["attention_mask"][..., 1:], prompt_inputs["attention_mask"]],
                                dim=-1,
                            ).squeeze(0)
                        )
                    else:
                        batch_input_ids.append(prompt_inputs["input_ids"].squeeze(0))
                        batch_attention_mask.append(prompt_inputs["attention_mask"].squeeze(0))

                # Now pad on the left to max length in the prompt, can't be customized by users for BC
                text_inputs = self.tokenizer.pad(
                    {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask},
                    padding_side="left",
                    max_length=None,
                )

            if images_replacements:
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        # Pop unused keys from the inputs, e.g. inputs used only to compute number of image tokens
        data = {**text_inputs, **processed_images}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}
        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs,
    ):
        # Don't call super on purpose, the model is extremely custom depending on inputs

        if text is not None and images is not None:
            prepared_text = []
            text = [text] if isinstance(text, str) else text
            images = [images] if not isinstance(images, (list, tuple)) else images
            for sample, image in zip(text, images):
                if not (isinstance(image, list) and image == []):
                    sample = self.image_token + sample

                # add eos tokens manually here, we'll add BOS in `replace_image_token`
                # because tokenizer's `_` ends up incorrect attached before BOS
                prepared_text.append(sample + BEGINNING_OF_ANSWER_STRING)
        elif text is not None:
            logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            prepared_text = text
        elif text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            prepared_text = [""]

        if images is not None:
            images = self.image_processor.fetch_images(images)
            images = make_flat_list_of_images(images)

        return images, prepared_text, None, None

    def validate_inputs(
        self,
        images: ImageInput | list[ImageInput] | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        super().validate_inputs(images=images, text=text, **kwargs)

        if text is None and images is None:
            raise ValueError("You must provide either `text` or `images`.")

    def _process_images(self, images: ImageInput, **kwargs):
        processed_images = self.image_processor(images, **kwargs)

        batch_image_patches = []
        image_replacements = []
        processor_output_images = processed_images.pop("images")
        for image_idx in range(len(images)):
            image_height, image_width = processor_output_images[image_idx].shape[-2:]
            patch_size = self.image_processor.patch_size
            patch_height, patch_width = patch_size.height, patch_size.width

            new_h = min(
                image_height,
                math.ceil(processed_images["image_unpadded_heights"][image_idx][0] / patch_height) * patch_height,
            )
            new_w = min(
                image_width,
                math.ceil(processed_images["image_unpadded_widths"][image_idx][0] / patch_width) * patch_width,
            )
            num_patches = self.image_processor.get_num_patches(
                image_height=new_h, image_width=new_w, patch_size=patch_size
            )
            image = processor_output_images[image_idx][..., :new_h, :new_w]
            image_patches = self.image_processor.patchify_image(image=image, patch_size=patch_size)
            batch_image_patches.append(image_patches.squeeze(0))

            row_width = new_w // patch_width
            replacement_text = self.replace_image_token(
                processed_images, image_idx=image_idx, num_patches=num_patches, row_width=row_width, **kwargs
            )
            image_replacements.append(replacement_text)

        processed_images["image_patches"] = torch.cat(batch_image_patches, dim=0)
        return processed_images, image_replacements

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # Terminate each line with newline token
        image_tokens_with_newlines = []
        for _ in range(0, kwargs["num_patches"], kwargs["row_width"]):
            image_tokens_with_newlines.append(self.image_token * kwargs["row_width"] + self.image_newline_token)

        # add BOS token after the image tokens, order kept for BC
        return "".join(image_tokens_with_newlines) + "<s>"

    @property
    def unused_input_names(self) -> list[str]:
        return ["image_unpadded_heights", "image_unpadded_widths", "image_sizes", "image_scale_factors"]

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.

        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.

        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            size = kwargs.get("size") or self.image_processor.size
            padded_height, padded_width = size["height"], size["width"]

            num_image_tokens = []
            num_image_patches = [1] * len(image_sizes)
            for image_size in image_sizes:
                height_scale_factor = padded_height / image_size[0]
                width_scale_factor = padded_width / image_size[1]
                optimal_scale_factor = min(height_scale_factor, width_scale_factor)

                image_unpadded_h = min(int(image_size[0] * optimal_scale_factor), image_size[0])
                image_unpadded_w = min(int(image_size[1] * optimal_scale_factor), image_size[1])

                # We can use torch here because Fuyu processor has hard dependency on torch. NOTE: Fuyu can't do multi-image
                # thus the below (1, 1, 1) is hardcoded. Same as when calling the processor
                model_image_input = self.image_processor.preprocess_with_tokenizer_info(
                    image_input=torch.zeros(1, 1, 3, padded_height, padded_width),
                    image_present=torch.ones(1, 1, 1),
                    image_unpadded_h=torch.tensor([[image_unpadded_h]]),
                    image_unpadded_w=torch.tensor([[image_unpadded_w]]),
                    image_placeholder_id=0,  # dummy ids, we can be sure `id=0` is never out-of-range
                    image_newline_id=0,
                    variable_sized=True,
                )
                num_image_tokens.append(model_image_input["image_input_ids"][0][0].shape[-1])
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        """
        Transforms raw coordinates detected by [`FuyuForCausalLM`] to the original images' coordinate space.
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

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-processes the output of `FuyuForConditionalGeneration` to only return the text output.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                containing the token ids of the generated sequences.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text output.
        """
        beginning_of_answer = self.tokenizer.convert_tokens_to_ids(BEGINNING_OF_ANSWER_STRING)
        # get boa index for each outputted sequence tensor
        # start all generated sequences from the beginning of the answer token, pad to have consistent length
        unpadded_output_sequences = [
            seq[(seq == beginning_of_answer).nonzero(as_tuple=True)[0] + 1 :] for seq in generated_outputs
        ]
        max_len = max(len(seq) for seq in unpadded_output_sequences)
        # convert to torch and pad sequences
        padded_output_sequences = torch.full((len(unpadded_output_sequences), max_len), self.tokenizer.pad_token_id)
        for i, seq in enumerate(unpadded_output_sequences):
            padded_output_sequences[i, : len(seq)] = torch.tensor(seq)

        return self.batch_decode(padded_output_sequences, skip_special_tokens=skip_special_tokens, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(tokenizer_input_names + ["image_patches"])


__all__ = ["FuyuProcessor"]
