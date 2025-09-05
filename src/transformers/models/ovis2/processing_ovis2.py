# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class Ovis2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "image_kwargs": {},
    }


class Ovis2Processor(ProcessorMixin):
    r"""
    Constructs a Ovis2 processor which wraps Ovis2 image processor and a Qwen2 tokenizer into a single processor.

    [`Ovis2Processor`] offers all the functionalities of [`Ovis2VideoProcessor`], [`Ovis2ImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Ovis2Processor.__call__`] and [`~Ovis2Processor.decode`] for more information.

    Args:
        image_processor ([`Ovis2ImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        image_seq_length (`int`, *optional*, defaults to 256):
            The number of image tokens to be used for each image in the input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        image_seq_length=256,
        **kwargs,
    ):
        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        **kwargs: Unpack[Ovis2ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        Ovis2ImageProcessor's [`~Ovis2ImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            Ovis2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
            image_grids = image_inputs.pop("grids").tolist()
            text = self._expand_image_tokens(text, image_grids)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs})

    def _expand_image_tokens(
        self,
        text: list[TextInput],
        grids: list[list[int]],
    ):
        processed_text = []
        grid_index = 0
        for sample in text:
            while "<image>" in sample:
                grid = grids[grid_index]
                row, col = grid[0], grid[1]
                placeholder = f"<IMG_START>{'<IMG_ATOM>' * self.image_seq_length}<IMG_GRID>"
                if row * col > 1:
                    for r in range(row):
                        for c in range(col):
                            placeholder += f"{'<IMG_ATOM>' * self.image_seq_length}"
                            if c < col - 1:
                                placeholder += "<IMG_COL>"
                        if r < row - 1:
                            placeholder += "<IMG_ROW>"
                placeholder += "<IMG_END>"

                sample = sample.replace("<image>", placeholder, 1)
                grid_index += 1
            processed_text.append(sample)
        return processed_text

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

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names) + list(image_processor_input_names)


__all__ = ["Ovis2Processor"]
