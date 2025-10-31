# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
Processor class for Bagel.
"""

# ToDo: see if a chat template is needed for Bagel or not once end-to-end model is ready.

from typing import Union

import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.\n\n"
)


class BagelTextKwargs(TextKwargs, total=False):
    generation_mode: str
    transforms_type: str  # Cleanup later as it's image kwargs


class BagelProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: BagelTextKwargs
    _defaults = {
        "text_kwargs": {"padding": False, "generation_mode": "text", "transforms_type": "vit"},
        "common_kwargs": {"return_tensors": "pt"},
    }


class BagelProcessor(ProcessorMixin):
    r"""
    Constructs a Bagel processor which wraps a Bagel Image Processor and a Qwen2 tokenizer into a single processor.

    [`BagelProcessor`] offers all the functionalities of [`BagelImageProcessor`] and [`Qwen2Tokenizer`]. See the
    [`~BagelProcessor.__call__`] and [`~BagelProcessor.decode`] for more information.

    Args:
        image_processor ([`BagelImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`Qwen2Tokenizer`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        use_default_system_prompt (`str`, *optional*, defaults to `False`):
            Use default system prompt for Text Generation.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BagelImageProcessor"
    tokenizer_class = "Qwen2TokenizerFast"

    def __init__(self, image_processor, tokenizer, chat_template=None, use_default_system_prompt=False, **kwargs):
        # Num image tokens is computed dynamically?
        self.num_image_tokens = 1
        self.image_start_token = tokenizer.boi_token
        self.image_end_token = tokenizer.eoi_token
        self.use_default_system_prompt = use_default_system_prompt

        # Dummy token for now
        self.image_token = "<|vision_pad|>"

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[BagelProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2Tokenizer's [`~Qwen2Tokenizer.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        BagelImageProcessor's [`~BagelImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
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

        output_kwargs = self._merge_kwargs(
            BagelProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        generation_mode = output_kwargs["text_kwargs"].pop("generation_mode")
        transforms_type = output_kwargs["text_kwargs"].pop("transforms_type")

        # Replace the image token with expanded image tokens.
        prompt_strings = []
        one_img_tokens = self.image_start_token + (self.image_token * self.num_image_tokens) + self.image_end_token

        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            if self.use_default_system_prompt and generation_mode == "text":
                prompt = DEFAULT_SYSTEM_PROMPT + prompt
            if generation_mode == "image":
                prompt += self.start_of_image
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # Process images if pixel values are provided.
        if images is not None and generation_mode != "image":
            data["pixel_values"] = self.image_processor(
                images=images, **output_kwargs["images_kwargs"], transforms_type=transforms_type
            )["pixel_values"]

        image_related_token_ids = set(
            self.tokenizer.convert_tokens_to_ids([self.image_start_token, self.image_end_token, self.image_token])
        )

        # token_type_ids: 1 for image tokens, 0 for text ones
        token_type_ids = []
        for input_ids in data["input_ids"]:
            type_ids = [1 if token_id in image_related_token_ids else 0 for token_id in input_ids.tolist()]
            token_type_ids.append(type_ids)

        data["token_type_ids"] = torch.tensor(token_type_ids)

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def postprocess(self, images: ImageInput, **kwargs):
        """
        Forwards all arguments to the image processor's `postprocess` method.
        Refer to the original method's docstring for more details.
        """
        return self.image_processor.postprocess(images, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["BagelProcessor"]
