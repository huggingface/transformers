# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for Molmo.
"""

from typing import List, Union, Optional

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, _validate_images_text_input_order, ImagesKwargs, TextKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging
import numpy as np

logger = logging.get_logger(__name__)

class MolmoImagesKwargs(ImagesKwargs, total=False):
    max_crops: Optional[int]
    overlap_margins: Optional[List[int]]
    base_image_input_size: Optional[List[int]]
    image_token_length_w: Optional[int]
    image_token_length_h: Optional[int]
    image_patch_size: Optional[int]
    image_padding_mask: Optional[bool]

class MolmoTextKwargs(TextKwargs, total=False):
    style: Optional[str]
    system_prompt: Optional[str]
    message_format: Optional[str]
    always_start_with_space: Optional[bool]
    sequence_length: Optional[int]


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: MolmoTextKwargs
    images_kwargs: MolmoImagesKwargs
    _defaults = {
        "images_kwargs": {
            "max_crops": 12,
            "overlap_margins": (4, 4),
            "tokens_per_image_width": 12,
            "tokens_per_image_height": 12,
            "image_patch_size": 14,
            "image_padding_mask": True,
        },
        "text_kwargs": {
            "padding": False,
        },
    }

DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"

class MolmoProcessor(ProcessorMixin):
    r"""
    Constructs a Molmo processor which wraps a Molmo image processor and a Molmo tokenizer into a single processor.

    [`MolmoProcessor`] offers all the functionalities of [`MolmoImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MolmoProcessor.__call__`] and [`~MolmoProcessor.decode`] for more information.

    Args:
        image_processor ([`MolmoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<|image|>",  # set the default and let users change if they have peculiar special tokens in rare cases
        **kwargs,
    ):
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MolmoImageProcessor's [`~MolmoImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        # TODO should be vectorizable
        if image_inputs.get("pixel_values") is not None and image_inputs.get("crop_grids") is not None:
            if self.patch_size is not None:
                for crop_grid, patch_ordering in zip(image_inputs.get("crop_grids"), image_inputs.get("patch_orderings")):
                    overlap_margins = self.image_processor.overlap_margins
                    crop_window_patches = self.image_processor.crop_window_patches


                    full_height = crop_grid[0] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                    full_width = crop_grid[1] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                    tokens_per_row = np.full(( (full_width + 1) // 2,), DEFAULT_IMAGE_PATCH_TOKEN, )
                    tokens_per_row = np.concatenate([tokens_per_row, [DEFAULT_IM_COL_TOKEN]], 0)

                    crop_tokens = np.tile(tokens_per_row, [(full_height + 1) // 2])
                    crop_tokens = [
                        [DEFAULT_IM_START_TOKEN],
                        crop_tokens,
                        [DEFAULT_IM_END_TOKEN]
                    ]

                    # for the global image

                    global_tokens_per_row = np.full(
                        (self.image_processor.tokens_per_image_width,),
                        DEFAULT_IMAGE_PATCH_TOKEN,
                    )
                    global_tokens_per_row = np.concatenate([global_tokens_per_row, [DEFAULT_IM_COL_TOKEN]], 0)
                    extra_tokens = np.tile(global_tokens_per_row, [self.image_processor.tokens_per_image_height])
                    all_image_tokens = [
                                [DEFAULT_IM_START_TOKEN],
                                extra_tokens,
                                [DEFAULT_IM_END_TOKEN],
                            ] + crop_tokens

                    all_image_tokens = np.concatenate(all_image_tokens, 0)

                    # then build the image token indices with the patch ordering baked in

                    image_token_mask = np.nonzero(all_image_tokens == DEFAULT_IMAGE_PATCH_TOKEN)[0].astype(np.int32)
                    number_of_tokens = image_token_mask.shape[0]
                    patch_ordering = np.reshape(patch_ordering, [-1])
                    valid = patch_ordering >= 0
                    number_of_valid_patches = valid.sum()

                    sorted_patch_ixs = np.zeros([number_of_tokens], np.int32)
                    sorted_patch_ixs[patch_ordering[valid]] = np.arange(number_of_valid_patches, dtype=np.int32)

                    # Project the inverted mapping into same sparse structure
                    sorted_patch_ixs_ex = np.full(np.shape(patch_ordering), -1)
                    sorted_patch_ixs_ex[valid] = sorted_patch_ixs

                    # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
                    valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
                    image_token_mask = image_token_mask[sorted_patch_ixs_ex * valid]
                    image_token_mask = image_token_mask * valid - 100 * (1 - valid)
                    image_token_mask = np.reshape(image_token_mask, [-1, self.image_processor.tokens_per_image_width * self.image_processor.tokens_per_image_height])
                    # Replace the image token with the expanded image token sequence
                    prompt_strings = []
                    for sample in text:
                        sample = sample.replace(self.image_token, "".join(all_image_tokens))
                        prompt_strings.append(sample)
            else:
                logger.warning_once(
                    "Expanding inputs for image tokens in Molmo should be done in processing. "
                    "Please add `patch_size` and to the model's processing config or set directly "
                    "with `processor.patch_size = {{patch_size}}`. "
                )

        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs})

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
