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
Processor class for Mllama.
"""

# TODO: update all docs

from typing import List, Optional, Union

# TODO: uncomment
# try:
#     from typing import Unpack
# except ImportError:
from typing_extensions import Unpack

from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
)
from ...tokenization_utils_base import (
    BatchEncoding,
    PreTokenizedInput,
    TextInput,
)

# TODO: Can we do it that way or its better include as "Copied from ..."
from .image_processing_mllama import make_list_of_images


class MllamaImagesKwargs(ImagesKwargs, total=False):
    max_image_tiles: Optional[int]


class MllamaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MllamaImagesKwargs

    _defaults = {
        "image_kwargs": {
            "max_image_tiles": 4,
        },
    }


class MllamaProcessor(ProcessorMixin):
    r"""
    Constructs a Mllama processor which wraps [`MllamaImageProcessor`] and
    [`PretrainedTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities. See the [`~MllamaProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more
    information.
    The preferred way of passing kwargs is as a dictionary per modality, see usage example below.
        ```python
        from transformers import MllamaProcessor
        from PIL import Image

        # TODO: fill model_id
        model_id = ""
        processor = MllamaProcessor.from_pretrained(model_id)

        processor(
            images=your_pil_image,
            text=["What is that?"],
            images_kwargs = {"size": {"height": 224, "width": 224}},
            text_kwargs = {"padding": "left"},
            common_kwargs = {"return_tensors": "pt"},
        )
        ```

    Args:
        image_processor ([`EfficientNetImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`BertTokenizer`, `BertTokenizerFast`]):
            The tokenizer is a required input.

    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "MllamaImageProcessor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, image_processor, tokenizer):
        self.image_token = "<|image|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.python_token = "<|python_tag|>"
        self.python_token_id = tokenizer.convert_tokens_to_ids(self.python_token)
        self.chat_template = tokenizer.chat_template
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        images: Optional[ImageInput] = None,
        **kwargs: Unpack[MllamaProcessorKwargs],
    ) -> BatchEncoding:
        """
        Main method to prepare text(s) and image(s) to be fed as input to the model. This method forwards the `text`
        arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` arguments to
        MllamaImageProcessor's [`~MllamaImageProcessor.__call__`] if `images` is not `None`. Please refer
        to the docstring of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
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
        if text is None and images is None:
            raise ValueError("You must specify either text or images.")

        output_kwargs = self._merge_kwargs(
            MllamaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        images_kwargs = output_kwargs["images_kwargs"]
        common_kwargs = output_kwargs["common_kwargs"]

        # remove the return_tensors key modality kwargs
        text_kwargs.pop("return_tensors", None)
        images_kwargs.pop("return_tensors", None)

        data = {}

        # for data that can't be represented as tensors,
        # because it stores nested objects of variable length
        not_tensor_data = {}

        if text is not None:

            if isinstance(text, str):
                text = [text]
            elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
                raise ValueError(
                    "Invalid input text. Please provide a string, or a list of strings"
                )
            n_images_in_text = [t.count(self.image_token) for t in text]
            encoding = self.tokenizer(text, **text_kwargs)
            data.update(encoding)

        if images is not None:

            images = make_list_of_images(images)
            n_images_in_images = [len(sample) for sample in images]

            if text is not None and not n_images_in_images == n_images_in_text:
                raise ValueError(
                    f"The number of images in the text {n_images_in_text} and images  {n_images_in_images} should be the same."
                )

            image_features = self.image_processor(images, **images_kwargs)
            not_tensor_data["num_tiles"] = image_features.pop("num_tiles")
            data.update(image_features)

        return_tensors = common_kwargs.pop("return_tensors", None)
        batch_encoding = BatchEncoding(data=data, tensor_type=return_tensors, **common_kwargs)
        batch_encoding.update(not_tensor_data)

        # fill missing keys with None
        for key in self.model_input_names:
            if key not in batch_encoding:
                batch_encoding[key] = None

        return batch_encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(tokenizer_input_names + image_processor_input_names + ["cross_attention_token_mask"])

    def cross_attention_token_mask(
        self, input_ids: Union[List[int], List[List[int]]]
    ) -> Union[List[List[int]], List[List[List[int]]]]:
        if input_ids and isinstance(input_ids[0], list):
            return [self.cross_attention_token_mask(t) for t in input_ids]

        image_token_locations = [i for i, token in enumerate(input_ids) if token == self.image_token_id]

        if len(image_token_locations) == 0:
            return []

        # only one image present, unmask until end of sequence
        if len(image_token_locations) == 1:
            return [[image_token_locations[0], -1]]

        vision_masks = [[loc1, loc2] for loc1, loc2 in zip(image_token_locations[:-1], image_token_locations[1:])]

        # last image will attend to all subsequent text
        vision_masks.append([image_token_locations[-1], len(input_ids)])

        # if there are two or more consecutive vision tokens,
        # they should all attend to all subsequent
        # text present
        last_mask_end = vision_masks[-1][1]
        for vision_mask in vision_masks[::-1]:
            if vision_mask[0] == vision_mask[1] - 1:
                vision_mask[1] = last_mask_end
            last_mask_end = vision_mask[1]

        return vision_masks
