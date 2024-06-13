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
Processor class for BLIP-2.
"""

from typing import List, Optional, Union

from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


class Blip2Processor(ProcessorMixin):
    r"""
    Constructs a BLIP-2 processor which wraps a BLIP image processor and an OPT/T5 tokenizer into a single processor.

    [`BlipProcessor`] offers all the functionalities of [`BlipImageProcessor`] and [`AutoTokenizer`]. See the docstring
    of [`~BlipProcessor.__call__`] and [`~BlipProcessor.decode`] for more information.

    Args:
        image_processor (`BlipImageProcessor`):
            An instance of [`BlipImageProcessor`]. The image processor is a required input.
        tokenizer (`AutoTokenizer`):
            An instance of ['PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.__init__
    def __init__(self, image_processor, tokenizer):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self.processing_kwargs: ProcessingKwargs = {
            "common_kwargs": {"return_tensors": None},
            "text_kwargs": {
                "add_special_tokens": True,
                "padding": False,
                "truncation": None,
                "max_length": None,
                "stride": 0,
                "pad_to_multiple_of": None,
                "return_attention_mask": None,
                "return_overflowing_tokens": False,
                "return_special_tokens_mask": False,
                "return_offsets_mapping": False,
                "return_token_type_ids": False,
                "return_length": False,
                "verbose": True,
            },
            "images_kwargs": {},
        }

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.__call__
    def __call__(
        self,
        images: ImageInput = None,
        text: Optional[Union[str, List[str], TextInput, PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`BlipImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        Args:
            images (`ImageInput`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
        """
        if images is None and text is None:
            raise ValueError("You have to specify either images or text.")

        text_encoding = None

        if text is not None:
            text_kwargs = {
                **self.processing_kwargs["text_kwargs"],
                **self.processing_kwargs["common_kwargs"],
                **kwargs,
            }
            text_encoding = self.tokenizer(text, **text_kwargs)

        # add pixel_values encoding. If we also have text_encoding, update image encoding and return it.
        # else, return the text encoding.

        if images is not None:
            images_kwargs = {
                **self.processing_kwargs["images_kwargs"],
                **self.processing_kwargs["common_kwargs"],
                **kwargs,
            }
            encoding_image_processor = self.image_processor(images, **images_kwargs)
            if text_encoding is not None:
                encoding_image_processor.update(text_encoding)
            return encoding_image_processor

        return text_encoding

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
