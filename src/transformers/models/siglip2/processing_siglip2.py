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
Image/Text processor class for SigLIP2.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class Siglip2ImagesKwargs(ImagesKwargs, total=False):
    max_num_patches: Optional[int]
    patch_size: Optional[int]


class Siglip2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Siglip2ImagesKwargs

    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
        },
        "images_kwargs": {
            "max_num_patches": 256,
            "patch_size": 16,
        },
    }


class Siglip2Processor(ProcessorMixin):
    r"""
    Constructs a Siglip2 processor which wraps a Siglip2 image processor and a Gemma tokenizer into a single processor.

    [`Siglip2Processor`] offers all the functionalities of [`Siglip2ImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~Siglip2Processor.__call__`] and [`~Siglip2Processor.decode`] for more information.

    Args:
        image_processor ([`Siglip2ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        images: Optional[Union[ImageInput, List[ImageInput], List[List[ImageInput]]]] = None,
        text: Optional[Union[TextInput, "PreTokenizedInput", List[TextInput], List["PreTokenizedInput"]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Siglip2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizerFast's [`~GemmaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` argument to
        Siglip2ImageProcessor's [`~Siglip2ImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `max_length`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*, defaults to 64):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*, defaults to `True`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'pt'`):
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
            - **pixel_attention_mask** -- Attention mask for the pixel values. Returned when `images` is not `None`.
            - **spatial_shapes** -- The number of horizontal and vertical patches per image.
              Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Siglip2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is not None and images is not None:
            encoding.update(image_features)
            return encoding
        elif text is not None:
            return encoding
        else:
            return_tensors = output_kwargs["common_kwargs"]["return_tensors"]
            return BatchFeature(data=dict(**image_features), tensor_type=return_tensors)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Siglip2Tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Siglip2Tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["Siglip2Processor"]
