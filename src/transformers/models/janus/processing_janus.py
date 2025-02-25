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
Processor class for Janus.
"""

from typing import List, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)
from ...utils import logging


logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image_placeholder>"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.\n\n"
)

# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)

class JanusProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {"padding": False, "return_tensors": "pt"},
        "common_kwargs": {"return_tensors": "pt"},
    }


class JanusProcessor(ProcessorMixin):
    r"""
    Constructs a Janus processor which wraps a Janus Image Processor and a Llama tokenizer into a single processor.

    [`JanusProcessor`] offers all the functionalities of [`JanusImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~JanusProcessor.__call__`] and [`~JanusProcessor.decode`] for more information.

    Args:
        image_processor ([`JanusImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "JanusImageProcessor"
    """
    The default from the original Janus codebase uses LlamaTokenizerFast, but not LlamaTokenizer.
    Trying to load their HUB tokenizer config with LlamaTokenizer.from_pretrained(model_path)
    throws an error due to the sentencepiece parameter not being found. Keeping the regular LlamaTokenizer here
    results in errors when testing, as the ProcessorTesterMixin.get_component() method tries to load the tokenizer
    using LlamaTokenizer.from_pretrained(model_path).
    """
    tokenizer_class = ("LlamaTokenizerFast")

    def __init__(self, image_processor, tokenizer, chat_template=None, use_default_system_prompt=True, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.num_image_tokens = 576
        self.image_start_token = "<begin_of_image>"  # Can be Hardcoded as it won't change.
        self.image_end_token = "<end_of_image>"
        self.use_default_system_prompt = use_default_system_prompt
        self.generation_mode = kwargs.get("generation_mode", None)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        **kwargs: Unpack[JanusProcessorKwargs],
    ) -> BatchFeature:
        """Construct a Janus processor with Janus Image procesor and Llama text tokenizer"""

        output_kwargs = self._merge_kwargs(
            JanusProcessorKwargs, tokenizer_init_kwargs=self.tokenizer.init_kwargs, **kwargs
        )

        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        if not self.generation_mode:
            logger.info("Generation mode argument not provided. Defaulting to 'text' mode.")
            self.generation_mode = "text"

        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # Replace the image token with expanded image tokens.
        prompt_strings = []
        one_img_tokens = self.image_start_token + (IMAGE_TOKEN * self.num_image_tokens) + self.image_end_token
        for prompt in text:
            prompt = prompt.replace(IMAGE_TOKEN, one_img_tokens)
            if self.use_default_system_prompt and self.generation_mode == "text":
                prompt = DEFAULT_SYSTEM_PROMPT + prompt
            if self.generation_mode == "image":
                prompt += self.image_start_token
            prompt_strings.append(prompt)


        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        # Process images if pixel values are provided.
        if images is not None and self.generation_mode != "image":
            data["pixel_values"] = self.image_processor(images=images, **output_kwargs["images_kwargs"])[
                "pixel_values"
            ]

        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["JanusProcessor"]
