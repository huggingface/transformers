# Copyright 2025 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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

import re
from typing import Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import logging


logger = logging.get_logger(__name__)


class DeepSeekOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


class DeepSeekOCRProcessor(ProcessorMixin):
    r"""
    Constructs a DeepSeek OCR processor which wraps an image processor and a tokenizer into a single processor.

    [`DeepSeekOCRProcessor`] offers all the functionalities of [`DeepSeekOCRImageProcessorFast`] and tokenizer.
    See the [`~DeepSeekOCRProcessor.__call__`] and [`~DeepSeekOCRProcessor.decode`] for more information.

    Args:
        image_processor (`DeepSeekOCRImageProcessorFast`):
            The image processor to use for images.
        tokenizer (PreTrainedTokenizer):
            The tokenizer to use for text.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The image token to use.
    """

    attributes = ["image_processor", "tokenizer"]
    tokenizer_class = "AutoTokenizer"
    image_processor_class = "DeepSeekOCRImageProcessorFast"

    def __init__(
        self,
        image_processor,
        tokenizer,
        image_token="<image>",
        **kwargs,
    ):
        self.image_token = image_token
        super().__init__(image_processor, tokenizer, **kwargs)

    def __call__(
        self,
        text: Union[TextInput, list[TextInput]],
        images: Optional[ImageInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s).

        Args:
            text (`str`, `list[str]`):
                The sequence or batch of sequences to be encoded.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, etc.):
                The image or batch of images to be prepared.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model.
            - **image_attention_mask** -- Mask for image tokens in the input sequence.
            - **image_spatial_crop** -- Spatial crop information for images.
        """

        output_kwargs = self._merge_kwargs(DeepSeekOCRProcessorKwargs, self.tokenizer.init_kwargs, **kwargs)
        image_kwargs = output_kwargs["images_kwargs"]

        image_inputs = self.image_processor(images, **image_kwargs) if images is not None else {}

        num_img_tokens = image_inputs.pop("num_img_tokens", [])

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        concatenated_prompt = "".join(text)
        if concatenated_prompt.count(self.image_token) != len(num_img_tokens):
            raise ValueError(
                f"Number of image tokens ({concatenated_prompt.count(self.image_token)}) in text "
                f"does not match number of images ({len(num_img_tokens)}). "
                f"Please add {self.image_token} token for each image."
            )

        image_count_iter = iter(num_img_tokens)
        processed_text = [re.sub(re.escape(self.image_token), lambda _: self.image_token * next(image_count_iter), t) for t in text]

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(processed_text, **output_kwargs["text_kwargs"])

        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        input_ids = text_inputs["input_ids"]
        if isinstance(input_ids, list):
            batch_size = len(input_ids)
        else:
            batch_size = input_ids.size(0)

        import torch

        image_attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for batch_idx in range(batch_size):
            if isinstance(input_ids, list):
                ids = input_ids[batch_idx]
            else:
                ids = input_ids[batch_idx]

            image_positions = (ids == image_token_id).nonzero(as_tuple=True)[0]

            for pos in image_positions:
                image_attention_mask[batch_idx, pos] = True

        data = {
            **text_inputs,
            **image_inputs,
            "image_attention_mask": image_attention_mask,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + ["image_attention_mask"]))


__all__ = ["DeepSeekOCRProcessor"]
