import re
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

from ...image_utils import (
    ChannelDimension,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    to_numpy_array,
)
from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available, is_vision_available, logging


if is_torch_available() and is_vision_available():
    from .image_processing_fuyu import FuyuImageProcessor


logger = logging.get_logger(__name__)

if is_vision_available():
    import PIL

if is_torch_available():
    import torch

BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
POINT_OPEN_STRING = "<0x02>"  # <point>
POINT_CLOSE_STRING = "<0x03>"  # </point>

TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = BBOX_OPEN_STRING = "<0x00>"  # <bbox>
BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_BBOX_CLOSE_STRING = TOKEN_POINT_OPEN_STRING = POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>


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
    num_real_text_tokens: Union[List[List[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: List[List["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> List["torch.Tensor"]:
    """Takes an input_stream tensor of shape B x S x ?. For each subsequence, adds any required
    padding to account for images and then unpacks the subsequences to create a single sequence per item in the batch.
    Returns a list of tensors, one for each item in the batch."""

    all_bi_stream = []

    for bi in range(batch_size):
        all_si_stream = []

        # First, construct full token stream (including image placeholder tokens) and loss mask for each subsequence
        # and append to lists. We use lists rather than tensors because each subsequence is variable-sized.
        for si in range(num_sub_sequences):
            image_adjustment = image_tokens[bi][si]
            si_stream = torch.cat([image_adjustment, input_stream[bi, si]], dim=0)
            num_real_tokens = image_adjustment.shape[0] + num_real_text_tokens[bi][si]

            all_si_stream.append(si_stream[:num_real_tokens])
        # Combine all subsequences for this batch entry. Still using a list because each batch entry is variable-sized.
        all_bi_stream.append(torch.cat(all_si_stream, dim=0))

    return all_bi_stream


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


def _transform_coordinates_and_tokenize(prompt: str, transformed_image, tokenizer) -> List[int]:
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
            within_tag_tokenized = _transform_within_tags(elem[0], transformed_image, tokenizer)
            # Surround the text with the open and close tags
            transformed_prompt_tokens.extend(within_tag_tokenized)
        else:
            transformed_prompt_tokens.extend(tokenizer(elem[0], add_special_tokens=False).input_ids)
    return transformed_prompt_tokens


def _transform_within_tags(text: str, transformed_image, tokenizer) -> List[int]:
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
    transformed_images: Optional[List[List["torch.Tensor"]]],
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """

    # If not tool use, tranform the coordinates while tokenizing
    if transformed_images is not None:
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
        bos_token = tokenizer.vocab["<s>"]
    else:
        bos_token = tokenizer.vocab["|ENDOFTEXT|"]
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
            prompt_tokens.extend([tokenizer.vocab["|ENDOFTEXT|"]] * padding_size)

    # Now we are in a structured format, we can convert to tensors.
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)

    return prompts_tokens_tensor, prompts_length_tensor


def original_to_transformed_h_coords(self, original_coords):
    # apply crop
    cropped_coords = (
        self._clamp_coords(original_coords, min_value=self.crop_top, max_value=self.crop_bottom) - self.crop_top
    )
    # apply scale
    scaled_coords = self._scale_coords(cropped_coords, scale=self.scaled_h / self.original_h)
    # apply pad
    return scaled_coords + self.padding_top


def original_to_transformed_w_coords(self, original_coords):
    # apply crop
    cropped_coords = (
        self._clamp_coords(original_coords, min_value=self.crop_left, max_value=self.crop_right) - self.crop_left
    )
    # apply scale
    scaled_coords = self._scale_coords(cropped_coords, scale=self.scaled_w / self.original_w)
    # apply pad
    return scaled_coords + self.padding_left


def scale_point_to_transformed_image(x: float, y: float) -> List[int]:
    x_scaled = original_to_transformed_w_coords(np.array([x / 2]))[0]
    y_scaled = original_to_transformed_h_coords(np.array([y / 2]))[0]
    return [x_scaled, y_scaled]


def scale_bbox_to_transformed_image(top: float, left: float, bottom: float, right: float) -> List[int]:
    top_scaled = original_to_transformed_w_coords(np.array([top / 2]))[0]
    left_scaled = original_to_transformed_h_coords(np.array([left / 2]))[0]
    bottom_scaled = original_to_transformed_w_coords(np.array([bottom / 2]))[0]
    right_scaled = original_to_transformed_h_coords(np.array([right / 2]))[0]
    return [top_scaled, left_scaled, bottom_scaled, right_scaled]


# Copied from transformers.models.detr.image_processing_detr.max_across_indices
def max_across_indices(values: Iterable[Any]) -> List[Any]:
    """
    Return the maximum value across all indices of an iterable of values.
    """
    return [max(values_i) for values_i in zip(*values)]


# Copied from transformers.models.detr.image_processing_detr.get_max_height_width
def get_max_height_width(
    images: List[np.ndarray], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> List[int]:
    """
    Get the maximum height and width across all images in a batch.
    """
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])

    if input_data_format == ChannelDimension.FIRST:
        _, max_height, max_width = max_across_indices([img.shape for img in images])
    elif input_data_format == ChannelDimension.LAST:
        max_height, max_width, _ = max_across_indices([img.shape for img in images])
    else:
        raise ValueError(f"Invalid channel dimension format: {input_data_format}")
    return (max_height, max_width)


# Copied from transformers.models.detr.image_processing_detr.make_pixel_mask
def make_pixel_mask(
    image: np.ndarray, output_size: Tuple[int, int], input_data_format: Optional[Union[str, ChannelDimension]] = None
) -> np.ndarray:
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`np.ndarray`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """
    input_height, input_width = get_image_size(image, channel_dim=input_data_format)
    mask = np.zeros(output_size, dtype=np.int64)
    mask[:input_height, :input_width] = 1
    return mask


class FuyuProcessor(ProcessorMixin):
    r"""
    Constructs a Fuyu processor which wraps a Fuyu image processor and a Llama tokenizer into a single processor.

    [`FuyuProcessor`] offers all the functionalities of [`FuyuImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~FuyuProcessor.__call__`] and [`~FuyuProcessor.decode`] for more information.

    Args:
        image_processor ([`FuyuImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "FuyuImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.image_processor = FuyuImageProcessor()

    def _process_images(self, images):
        """Utility function to preprocess the images and extract necessary information about original formats."""
        batch_images = []
        image_unpadded_heights = []
        image_unpadded_widths = []

        for image in images:
            image = to_numpy_array(image)
            if not is_scaled_image(image):
                image = image / 255.0
            channel_dimension = infer_channel_dimension_format(image, 3)
            if channel_dimension == ChannelDimension.FIRST:
                width_index = 2
                height_index = 1
            elif channel_dimension == ChannelDimension.LAST:
                width_index = 1
                height_index = 0

            image_unpadded_widths.append([image.shape[width_index]])
            image_unpadded_heights.append([image.shape[height_index]])

            # Reproduct adept padding sampler
            padded_image = self.image_processor.apply_transformation(image)

            tensor_img = torch.Tensor(padded_image).permute(2, 0, 1)
            batch_images.append([tensor_img])

        return batch_images, torch.Tensor(image_unpadded_heights), torch.Tensor(image_unpadded_widths)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
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

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]
            batch_images = []
            if isinstance(images, PIL.Image.Image):
                images = [images]
            if isinstance(images, list):
                batch_images, image_unpadded_heights, image_unpadded_widths = self._process_images(images)
                # image_unpadded_heights = image_unpadded_heights.unsqueeze(0)
                # image_unpadded_widths = image_unpadded_widths.unsqueeze(0)
            else:
                raise ValueError("images must be a list of ndarrays or PIL Images to be processed.")

            # Note: the original adept code has a handling of image_unpadded_h and w, but it doesn't seem to hold
            # when there are several different size subsequences per batch. The current implementation reflects
            # that limitation and should be documented.
            #
            self.subsequence_length = 1  # Each batch contains only one sequence.
            self.batch_size = len(batch_images)
            # FIXME max_tokens_to_generate is embedded into this processor's call.
            prompt_tokens, prompts_length = _tokenize_prompts_with_image_and_batch(
                tokenizer=self.tokenizer,
                prompts=prompts,
                transformed_images=batch_images,
                max_tokens_to_generate=self.max_tokens_to_generate,
                max_position_embeddings=self.max_position_embeddings,
                add_BOS=True,
                add_beginning_of_answer_token=True,
            )
            # same so far

            # This is 1 if there is an image per subsequence, else 0. [batch, 1, presence]
            # the remainder of current image processing logic assumes subsequence_size = 1.
            # Here it is OK as the model cannot handle > 1 subsequences
            # the image could be absent however and image presence should be inferred from user batch input
            # hence this code assumes the images are present. Use an assert?

            image_present = torch.ones(self.batch_size, 1, 1)

            image_placeholder_id = self.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
            image_newline_id = self.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
            tensor_batch_images = torch.stack([img[0] for img in batch_images]).unsqueeze(1)
            model_image_input = self.image_processor.process_images_for_model_input(
                image_input=tensor_batch_images,
                image_present=image_present,
                image_unpadded_h=image_unpadded_heights,
                image_unpadded_w=image_unpadded_widths,
                image_patch_dim_h=30,
                image_patch_dim_w=30,
                image_placeholder_id=image_placeholder_id,
                image_newline_id=image_newline_id,
                variable_sized=True,
            )

            image_padded_unpacked_tokens = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=prompt_tokens,
                image_tokens=model_image_input["image_input_ids"],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            # Construct inputs for image patch indices.
            unpacked_image_patch_indices_per_batch = construct_full_unpacked_stream(
                num_real_text_tokens=prompts_length,
                input_stream=torch.full_like(prompt_tokens, -1),
                image_tokens=model_image_input["image_patch_indices_per_batch"],
                batch_size=self.batch_size,
                num_sub_sequences=self.subsequence_length,
            )
            max_prompt_length = max(x.shape[-1] for x in image_padded_unpacked_tokens)
            max_seq_len_batch = min(max_prompt_length + self.max_tokens_to_generate, self.max_position_embeddings)
            all_bi_tokens_to_place = []
            for bi in range(self.batch_size):
                tokens_to_place = min(max_seq_len_batch, max(0, image_padded_unpacked_tokens[bi].shape[0]))
                all_bi_tokens_to_place.append(tokens_to_place)

            # Use same packing logic for the image patch indices.
            image_patch_input_indices = full_unpacked_stream_to_tensor(
                all_bi_tokens_to_place=all_bi_tokens_to_place,
                full_unpacked_stream=unpacked_image_patch_indices_per_batch,
                fill_value=-1,
                batch_size=self.batch_size,
                new_seq_len=max_seq_len_batch,
                offset=0,
            )

            image_patches_tensor = torch.stack([img[0] for img in model_image_input["image_patches"]]).unsqueeze(1)
            return {
                "input_ids": image_padded_unpacked_tokens[0].unsqueeze(0),
                "image_patches": image_patches_tensor[0][0].unsqueeze(0),
                "image_patches_indices": image_patch_input_indices,
            }

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)
