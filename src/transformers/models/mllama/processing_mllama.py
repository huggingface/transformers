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


class MllamaImagesKwargs(ImagesKwargs, total=False):
    do_image_splitting: Optional[bool]
    max_image_splits: Optional[int]


class MllamaProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "image_kwargs": {
            "do_image_splitting": True,
            "max_image_splits": 4,
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
        self.vision_token = "<|image|>"
        self.vision_token_id = tokenizer.convert_tokens_to_ids(self.vision_token)

        # TODO: this need resize_model_embeddings, original pad_id = -1, we cant do that
        # TODO: similar to Llama3, see Tips here https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        self.pad_token = "<|pad|>"
        tokenizer.add_special_tokens({"pad_token": self.pad_token})
        self.pad_token_id = tokenizer.convert_tokens_to_ids(self.pad_token)

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

        data = {}

        if text is not None:
            encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])
            data.update(encoding)

            # create mask for vision tokens
            # TODO: not working for return_tensors="pt"
            # Is it worth creating MllamaTokenizer for this method?
            vision_mask = self.create_vision_mask(encoding["input_ids"])
            data["vision_mask"] = vision_mask

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            data.update(image_features)

        return_tensors = output_kwargs["common_kwargs"].pop("return_tensors", None)
        batch_encoding = BatchEncoding(data=data, tensor_type=return_tensors)

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
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def create_vision_mask(
        self, tokens: Union[List[int], List[List[int]]]
    ) -> Union[List[List[int]], List[List[List[int]]]]:
        if tokens and isinstance(tokens[0], list):
            return [self.create_vision_mask(t) for t in tokens]

        vision_token_locations = [i for i, token in enumerate(tokens) if token == self.vision_token_id]
        if len(vision_token_locations) == 0:
            return []

        last_not_pad_token_id = max([i for i, token in enumerate(tokens) if token != self.pad_token_id])

        vision_masks = [[loc1, loc2] for loc1, loc2 in zip(vision_token_locations[:-1], vision_token_locations[1:])]

        # last image will attend to all subsequent text
        vision_masks.append([vision_token_locations[-1], last_not_pad_token_id + 1])

        # if there are two or more consecutive vision tokens,
        # they should all attend to all subsequent
        # text present
        last_mask_end = vision_masks[-1][1]
        for vision_mask in vision_masks[::-1]:
            if vision_mask[0] == vision_mask[1] - 1:
                vision_mask[1] = last_mask_end
            last_mask_end = vision_mask[1]

        return vision_masks

    # TODO: how to find total_len? its min(params.max_seq_len, max_gen_len + max_prompt_len)
    # TODO: worth it implement in modeling code?
    # max_prompt_len - OK, derived from tokens, can be computed in model or in processor
    # max_gen_len - ? stored in model config
    # params.max_seq_len - ? stored in model config
    # all_masks - OK, output of processor ("vision_mask")
    # all_num_chunks - OK, output of processor ("num_patches")
    # max_num_chunks - ? stored in image processor config, but can be derived from "pixel_values"
    @staticmethod
    def _pad_masks(
        all_masks: List[List[List[int]]],
        all_num_chunks: List[List[int]],
        total_len: int,
        max_num_chunks: int,
    ):
        import torch

        dtype = torch.bfloat16
        inf_value = float("-inf")

        bsz = len(all_masks)
        max_num_media = max([len(m) for m in all_masks])

        out_masks = torch.full(
            (bsz, total_len, max_num_media, max_num_chunks),
            inf_value,
            dtype=dtype,
        )

        for idx, (mask, num_chunks) in enumerate(zip(all_masks, all_num_chunks)):
            for mask_idx, (mask_elem, mask_num_chunks) in enumerate(zip(mask, num_chunks)):
                if len(mask_elem) == 2:
                    mask_elem[1] = min(mask_elem[1], total_len)
                    if mask_elem[1] == -1:
                        mask_elem[1] = total_len
                    out_masks[idx, mask_elem[0] : mask_elem[1], mask_idx, :mask_num_chunks].fill_(0.0)

        return out_masks
