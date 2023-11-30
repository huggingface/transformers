# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processor class for Llava.
"""

from typing import Callable, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, AddedToken
from ...utils import TensorType, is_torch_available


if is_torch_available():
    import torch

class LlavaProcessor(ProcessorMixin):
    r"""
    Constructs a Llava processor which wraps a LLama tokenizer and Llava image processor into a single processor.

    [`LlavaProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`LlamaTokenizerFast`]. See
    the docstring of [`~LlavaProcessor.__call__`] and [`~LlavaProcessor.decode`] for more information.

    Args:
        image_processor (`CLIPImageProcessor`):
            An instance of [`CLIPImageProcessor`]. The image processor is a required input.
        tokenizer (`LlamaTokenizerFast`, *optional*):
            An instance of [`LlamaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = "LlamaTokenizerFast"

    def __init__(self, image_processor, tokenizer=None):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        tokenizer.add_tokens(AddedToken("<image>", special = True, normalized = False))
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor

    def __call__(
        self,
        text=None,
        images=None,
        transform: Callable = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchEncoding:
        """This method takes batched or non-batched text made of text and images and converts them into text that
        the model was trained on and prepares the image pixel values for the model to process.

        Args:
            text (`Union[List[TextInput], [List[List[TextInput]]]]`):
                either a single prompt or a batched list of text - see the detailed description immediately after
                the end of the arguments doc section.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            transform (`Callable`, *optional*):
                A custom transform function that accepts a single image can be passed for training. For example,
                `torchvision.Compose` can be used to compose multiple functions. If `None` a preset inference-specific
                set of transforms will be applied to the images
            add_eos_token (`bool`, *optional*, defaults to `False`):
                Adds `eos_token` at the end of the final prompt if True`
            add_end_of_utterance_token (`bool`, *optional*)
                Whether to automatically add `<end_of_utterance>` after each prompt's text input (unless followed by an
                image). If `None` the tokenizer will be checked instead and if this token is found in
                `additional_special_tokens` then the value will be `True`.
            debug (`bool`, *optional*, defaults to `False`):
                `True` value will help debug prompt generation by dumping useful information
            return_tensors (`str` or `TensorType`, *optional*, defaults to `TensorType.PYTORCH`):
                The type of tensors to return. Can be one of:
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.

        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

        Detailed explanation:

        Each entry in `text` is either a text to be passed as is or an image that will be processed.

        An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.
        """
        if images is not None:
            pixel_values = self.image_processor(images, transform=transform, return_tensors=return_tensors)[
                "pixel_values"
            ]
        else:
            pixel_values = None

        # Attention mask have to be created later on? Or not?
        text_inputs = self.tokenizer(text, return_tensors=return_tensors)

        return BatchFeature(data={**text_inputs,"pixel_values": pixel_values})

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
