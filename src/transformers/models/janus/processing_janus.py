# coding=utf-8
# Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.
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
Processor class for Janus.
"""

from typing import Optional, Union

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


class JanusTextKwargs(TextKwargs, total=False):
    generation_mode: str


class JanusProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: JanusTextKwargs
    _defaults = {
        "text_kwargs": {"padding": False, "generation_mode": "text"},
        "common_kwargs": {"return_tensors": "pt"},
    }


class JanusProcessor(ProcessorMixin):
    r"""
    Constructs a Janus processor which wraps a Janus Image Processor and a Llama tokenizer into a single processor.

    [`JanusProcessor`] offers all the functionalities of [`JanusImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~JanusProcessor.__call__`] and [`~JanusProcessor.decode`] for more information.

    Args:
        image_processor ([`JanusImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`]):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        use_default_system_prompt (`str`, *optional*, defaults to `False`):
            Use default system prompt for Text Generation.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "JanusImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer, chat_template=None, use_default_system_prompt=False, **kwargs):
        self.num_image_tokens = 576
        self.image_token = tokenizer.image_token
        self.image_start_token = tokenizer.boi_token
        self.image_end_token = tokenizer.eoi_token
        self.use_default_system_prompt = use_default_system_prompt

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        images: Optional[ImageInput] = None,
        videos=None,
        audio=None,
        **kwargs: Unpack[JanusProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        JanusImageProcessor's [`~JanusImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
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
            JanusProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        generation_mode = output_kwargs["text_kwargs"].pop("generation_mode")

        # Replace the image token with expanded image tokens.
        prompt_strings = []
        one_img_tokens = self.image_start_token + (self.image_token * self.num_image_tokens) + self.image_end_token
        for prompt in text:
            prompt = prompt.replace(self.image_token, one_img_tokens)
            if self.use_default_system_prompt and generation_mode == "text":
                prompt = DEFAULT_SYSTEM_PROMPT + prompt
            if generation_mode == "image":
                prompt += self.image_start_token
            prompt_strings.append(prompt)

        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # Process images if pixel values are provided.
        if images is not None and generation_mode != "image":
            data["pixel_values"] = self.image_processor(images=images, **output_kwargs["images_kwargs"])[
                "pixel_values"
            ]

        return BatchFeature(data=data)

    def postprocess(self, images: ImageInput, **kwargs):
        """
        Forwards all arguments to the image processor's `postprocess` method.
        Refer to the original method's docstring for more details.
        """
        return self.image_processor.postprocess(images, **kwargs)


__all__ = ["JanusProcessor"]
