# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Emu3.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class Emu3TextKwargs(TextKwargs, total=False):
    return_for_image_generation: bool


class Emu3ImageKwargs(TextKwargs, total=False):
    ratio: str
    image_area: int


class Emu3ProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: Emu3TextKwargs
    _defaults = {
        "text_kwargs": {
            "return_for_image_generation": False,
        },
        "images_kwargs": {
            "ratio": "1:1",
            "image_area": 518400,
        },
    }


class Emu3Processor(ProcessorMixin):
    r"""
    Constructs a Emu3 processor which wraps a Emu3 image processor and a GPT2 tokenizer into a single
    processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3ImageProcessor`] and [`GPT2TokenizerFast`].
    See the [`~Emu3Processor.__call__`] and [`~Emu3Processor.decode`] for more information.

    Args:
        image_processor ([`Emu3ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`Emu3TokenizerFast`]):
            The tokenizer is a required input.
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast")
    valid_kwargs = ["image_seq_length", "image_token"]
    image_processor_class = "Emu3ImageProcessor"

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_seq_length: int = 1024,
        image_token: str = "<image>",
        chat_template=None,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token = image_token
        self.image_start_token = "<|image start|>"  # fixed tokens for start and end, so can hardcode
        self.image_end_token = "<|image end|>"
        self.fake_token_around_image = "<|image token|>"
        self.eol_token = ("<|extra_200|>",)
        self.eof_token = ("<|extra_201|>",)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[Emu3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3TokenizerFast's [`~Emu3TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
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
        # check if images and text inputs are reversed for BC

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        output_kwargs = self._merge_kwargs(
            Emu3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_for_image_generation = output_kwargs["text_kwargs"].pop("return_for_image_generation", False)
        ratio = output_kwargs["images_kwargs"].pop("ratio", None)
        image_area = output_kwargs["images_kwargs"].pop("image_area", None)

        if return_for_image_generation and images is not None:
            raise ValueError("You should not provide `images` when `return_for_image_generation=True`")

        if not return_for_image_generation and text is None and images is None:
            raise ValueError("You must provide either text or images when `return_for_image_generation=False`")

        # Replace the image token with the expanded image token sequence
        image_placeholder = f"{self.image_start_token}{self.fake_token_around_image}"
        if not return_for_image_generation:
            image_placeholder += (
                f"{self.image_token * self.image_seq_length}{self.eol_token}{self.eof_token}{self.image_end_token}"
            )

        prompt_strings = []
        for sample in text:
            sample = sample.replace(self.image_token, image_placeholder)
            prompt_strings.append(sample)
        data = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])

        if images is not None:
            data["pixel_values"] = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        return BatchFeature(data=data, tensor_type=output_kwargs["common_kwargs"]["return_tensors"])

    def postprocess(self, images: ImageInput, **kwargs):
        self.image_processor.postprocess(images, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Emu3TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Emu3TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
