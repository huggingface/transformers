# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
Processor class for BridgeTower.
"""

from typing import List, Union

from ...image_utils import ChannelDimension, ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


class BridgeTowerProcessor(ProcessorMixin):
    r"""
    Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single
    processor.

    [`BridgeTowerProcessor`] offers all the functionalities of [`BridgeTowerImageProcessor`] and
    [`RobertaTokenizerFast`]. See the docstring of [`~BridgeTowerProcessor.__call__`] and
    [`~BridgeTowerProcessor.decode`] for more information.

    Args:
        image_processor (`BridgeTowerImageProcessor`):
            An instance of [`BridgeTowerImageProcessor`]. The image processor is a required input.
        tokenizer (`RobertaTokenizerFast`):
            An instance of ['RobertaTokenizerFast`]. The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BridgeTowerImageProcessor"
    tokenizer_class = ("RobertaTokenizer", "RobertaTokenizerFast")

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)
        self.processing_kwargs: ProcessingKwargs = {
            "common_kwargs": {"return_tensors": None},
            "text_kwargs": {
                "add_special_tokens": True,
                "padding": False,
                "truncation": None,
                "max_length": None,
                "stride": 0,
                "is_split_into_words": False,
                "pad_to_multiple_of": None,
                "return_token_type_ids": None,
                "return_attention_mask": None,
                "return_overflowing_tokens": False,
                "return_special_tokens_mask": False,
                "return_offsets_mapping": False,
                "return_length": False,
                "verbose": True,
            },
            "images_kwargs": {
                "do_resize": None,
                "size": None,
                "size_divisor": None,
                "resample": None,
                "do_rescale": None,
                "rescale_factor": None,
                "do_normalize": True,
                "image_mean": None,
                "image_std": None,
                "do_pad": None,
                "pad_and_return_pixel_mask": None,
                "do_center_crop": True,
                "data_format": ChannelDimension.FIRST,
                "input_data_format": None,
            },
        }

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs,
    ) -> BatchEncoding:
        """
        This method uses [`BridgeTowerImageProcessor.__call__`] method to prepare image(s) for the model, and
        [`RobertaTokenizerFast.__call__`] to prepare text for the model.

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
        text_kwargs = {**self.processing_kwargs["text_kwargs"], **self.processing_kwargs["common_kwargs"], **kwargs}
        images_kwargs = {
            **self.processing_kwargs["images_kwargs"],
            **self.processing_kwargs["common_kwargs"],
            **kwargs,
        }

        if not text or not images:
            raise ValueError("Both `text` and `images` are expected as inputs.")
        if text:
            encoding = self.tokenizer(text, **text_kwargs)
        if images:
            image_features = self.image_processor(images, **images_kwargs)
            encoding.update(image_features)

        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to RobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
