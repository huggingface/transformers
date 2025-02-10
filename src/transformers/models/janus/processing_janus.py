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

from ...image_utils import is_valid_image
from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils import select_best_resolution
from ...image_utils import ImageInput, VideoInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging
from typing import TYPE_CHECKING, List, Optional, Union
from ...processing_utils import (
    ProcessorMixin,
)
from ...tokenization_utils_base import (
    AddedToken,
)
from ...utils import logging
import torch


logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image_placeholder>"

DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language.\n\n"
    )

# messages = [{"role":"User",
#   "content":[{'type':"text","text":"<image_placeholder>\nConvert the formula into latex code.\n"}]},
#   {"role": "Assistant", "content": " "},
# ]
# Here as  a hack I have added \n after user content but ideally chat template should add it

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
        "text_kwargs": {
            "padding": False,
            "return_tensors":"pt"
        }
    }

class JanusProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`JanusProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~JanusProcessor.__call__`] and [`~JanusProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "JanusImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    def __init__(self, image_processor, tokenizer, chat_template=None,use_default_system_prompt=True, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.num_image_tokens = 10 # revert back to 576 or fetch from pre_processor config
        self.image_start_token = "<begin_of_image>"
        self.image_end_token = "<end_of_image>"
        self.use_default_system_prompt = use_default_system_prompt

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: ImageInput = None,
        **kwargs: Unpack[JanusProcessorKwargs]
    ) -> BatchFeature:
        """Construct a Janus processor with JanusImage procesor and Llama text tokenizer"""

        output_kwargs = self._merge_kwargs(JanusProcessorKwargs,tokenizer_init_kwargs=self.tokenizer.init_kwargs,**kwargs)

        if text is None and images is None:
            raise ValueError("You must specify either text or images.")


        data = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # Replace the image token with explanded imaeg tokens.
        prompt_strings = []
        one_img_tokens = self.image_start_token + (IMAGE_TOKEN * self.num_image_tokens) + self.image_end_token
        for sample in text:
            sample = sample.strip()
            sample = sample.replace(IMAGE_TOKEN, one_img_tokens)
            if self.use_default_system_prompt:
                sample = DEFAULT_SYSTEM_PROMPT + sample
            prompt_strings.append(sample)


        data = self.tokenizer(prompt_strings,**output_kwargs['text_kwargs'])

        if images is not None:
            # How to pass image kwargs and it returns the pixel values aso append it to the output.
            data['pixel_values'] = self.image_processor(images=images,return_tensors="pt")['pixel_values']

        input_ids = data["input_ids"]
        batch_size, _ = input_ids.shape

        # Compute special tokens IDs
        image_token_id = self.tokenizer.vocab.get(IMAGE_TOKEN)
        image_start_id = self.tokenizer.vocab.get(self.image_start_token)

        # Compute image sequence mask
        images_seq_mask = (input_ids == image_token_id) | (input_ids == image_start_id)

        # Compute image embedding mask dynamically
        max_n_images = max(1,len(images))
        images_emb_mask = torch.zeros((batch_size, max_n_images, self.num_image_tokens + 1), dtype=torch.bool)

        for i in range(batch_size):
            img_positions = (input_ids[i] == image_start_id).nonzero(as_tuple=True)[0]
            for j, start_idx in enumerate(img_positions):
                end_idx = start_idx + self.num_image_tokens + 1  # Account for <image_beg>
                images_emb_mask[i, j, : min(end_idx - start_idx, self.num_image_tokens + 1)] = True

        # Process images if provided
        if images is not None:
            data["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]

        # Add masks to the output
        data.update({
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": images_emb_mask
        })

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