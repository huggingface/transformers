# coding=utf-8
# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang The HuggingFace Inc. team. All rights reserved.
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
Image/Text processor class for AltCLIP
"""
import warnings
from typing import List, Union

from ...image_utils import ChannelDimension, ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput


class AltCLIPProcessor(ProcessorMixin):
    r"""
    Constructs a AltCLIP processor which wraps a CLIP image processor and a XLM-Roberta tokenizer into a single
    processor.

    [`AltCLIPProcessor`] offers all the functionalities of [`CLIPImageProcessor`] and [`XLMRobertaTokenizerFast`]. See
    the [`~AltCLIPProcessor.__call__`] and [`~AltCLIPProcessor.decode`] for more information.

    Args:
        image_processor ([`CLIPImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        feature_extractor ([`CLIPFeatureExtractor`], *optional*):
            The feature extractor is a deprecated input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("XLMRobertaTokenizer", "XLMRobertaTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, feature_extractor=None):
        if "feature_extractor":
            warnings.warn(
                "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                " instead.",
                FutureWarning,
            )
        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

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
                "resample": None,
                "do_thumbnail": None,
                "do_align_long_axis": None,
                "do_pad": None,
                "do_rescale": None,
                "rescale_factor": None,
                "do_normalize": None,
                "image_mean": None,
                "image_std": None,
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
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to XLMRobertaTokenizerFast's [`~XLMRobertaTokenizerFast.__call__`] if `text` is not
        `None` to encode the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

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
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        text_kwargs = {**self.processing_kwargs["text_kwargs"], **self.processing_kwargs["common_kwargs"], **kwargs}
        images_kwargs = {
            **self.processing_kwargs["images_kwargs"],
            **self.processing_kwargs["common_kwargs"],
            **kwargs,
        }

        if text:
            encoding = self.tokenizer(text, **text_kwargs)
        if images:
            image_features = self.image_processor(images, **images_kwargs)
            encoding.update(image_features)

        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to XLMRobertaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to XLMRobertaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
