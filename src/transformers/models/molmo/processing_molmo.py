# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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


from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available


if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin

if is_torch_available():
    # Some fast processing utils depend on torch
    import torch

### PROCESSING CODE


class MolmoImagesKwargs(ImagesKwargs, total=False):
    device: Optional[str]
    max_num_crops: Optional[int]
    overlap_margins: Optional[tuple[int, int]]
    tokens_per_image_height: Optional[int]
    tokens_per_image_width: Optional[int]
    image_patch_size: Optional[int]
    image_padding_mask: Optional[bool]


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MolmoImagesKwargs
    _defaults = {
        "images_kwargs": {
            "max_num_crops": 12,
            "overlap_margins": (4, 4),
            "tokens_per_image_width": 12,
            "tokens_per_image_height": 12,
            "image_patch_size": 14,
            "image_padding_mask": True,
            "device": None,
        },
        "text_kwargs": {
            "padding": False,
        },
    }


class MolmoProcessor(ProcessorMixin):
    r"""
    Constructs a Molmo processor which wraps a Molmo image processor and a Molmo tokenizer into a single processor.

    [`MolmoProcessor`] offers all the functionalities of [`MolmoImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~MolmoProcessor.__call__`] and [`~MolmoProcessor.decode`] for more information.

    Args:
        image_processor ([`MolmoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = tokenizer.image_token
        self.boi_token = tokenizer.boi_token
        self.eoi_token = tokenizer.eoi_token
        self.im_patch_token = tokenizer.im_patch_token
        self.im_col_token = tokenizer.im_col_token
        self.bos_token = tokenizer.bos_token or tokenizer.eos_token

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        MolmoImageProcessor's [`~MolmoImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
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
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        # try to expand inputs in processing if we have the necessary parts
        prompt_strings = text
        # TODO should be vectorizable
        if image_inputs.get("pixel_values") is not None and image_inputs.get("crop_grids") is not None:
            for crop_grid, patch_ordering in zip(image_inputs.pop("crop_grids"), image_inputs.pop("patch_orderings")):
                overlap_margins = self.image_processor.overlap_margins
                crop_window_patches = self.image_processor.crop_window_patches
                if isinstance(crop_grid, torch.Tensor):
                    crop_grid = crop_grid.cpu().numpy()
                    patch_ordering = patch_ordering.cpu().numpy()
                full_height = crop_grid[0] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                full_width = crop_grid[1] * crop_window_patches + (overlap_margins[1] + overlap_margins[0])
                tokens_per_row = np.full(
                    ((full_width + 1) // 2,),
                    self.im_patch_token,
                )
                tokens_per_row = np.concatenate([tokens_per_row, [self.im_col_token]], 0)

                crop_tokens = np.tile(tokens_per_row, [(full_height + 1) // 2])
                crop_tokens = [[self.boi_token], crop_tokens, [self.eoi_token]]

                # for the global image

                global_tokens_per_row = np.full(
                    (self.image_processor.tokens_per_image_width,),
                    self.im_patch_token,
                )
                global_tokens_per_row = np.concatenate([global_tokens_per_row, [self.im_col_token]], 0)
                extra_tokens = np.tile(global_tokens_per_row, [self.image_processor.tokens_per_image_height])
                all_image_tokens = [
                    [self.boi_token],
                    extra_tokens,
                    [self.eoi_token],
                ] + crop_tokens
                all_image_tokens = np.concatenate(all_image_tokens, 0)

                # then build the image token indices with the patch ordering baked in

                image_token_mask = np.nonzero(all_image_tokens == self.im_patch_token)[0].astype(np.int32)
                number_of_tokens = image_token_mask.shape[0]

                patch_ordering = np.reshape(patch_ordering, [-1])
                valid = patch_ordering >= 0

                number_of_valid_patches = valid.sum()
                sorted_patch_ixs = np.zeros([number_of_tokens], np.int32)
                sorted_patch_ixs[patch_ordering[valid]] = np.arange(number_of_valid_patches, dtype=np.int32)

                # Project the inverted mapping into same sparse structure
                sorted_patch_ixs_ex = np.full(np.shape(patch_ordering), -1)
                sorted_patch_ixs_ex[valid] = sorted_patch_ixs

                # Do the gather and then re-masked outputs that were masked in `sorted_patch_ixs`
                valid = (sorted_patch_ixs_ex >= 0).astype(np.int32)
                image_token_mask = image_token_mask[sorted_patch_ixs_ex * valid]
                image_token_mask = image_token_mask * valid - 100 * (1 - valid)
                image_token_mask = np.reshape(
                    image_token_mask,
                    [-1, self.image_processor.tokens_per_image_width * self.image_processor.tokens_per_image_height],
                )
                image_inputs.setdefault("image_token_indices", []).append(image_token_mask)

                # Replace the image token with the expanded image token sequence
                prompt_strings = []
                for sample in text:
                    sample = sample.replace(self.image_token, "".join(all_image_tokens))
                    prompt_strings.append(sample)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"])
        # # shift patch mapping after addition of bos token (with left padding)
        # # necessary for batched generation
        bos_token_identifier = self.tokenizer.bos_token_id
        attention_mask = text_inputs["attention_mask"]
        input_sequences = text_inputs["input_ids"]
        use_numpy = isinstance(attention_mask, np.ndarray)

        if use_numpy:
            pad_positions = (attention_mask == 0).astype(np.int64)
            left_padding = np.cumprod(pad_positions, axis=-1).sum(axis=-1)
            first_valid_index = (attention_mask == 1).argmax(axis=-1)
            sample_indices = np.arange(input_sequences.shape[0])
            first_tokens = input_sequences[sample_indices, first_valid_index]
            bos_offsets = (
                (first_tokens == bos_token_identifier).astype(np.int64) if bos_token_identifier is not None else 0
            )
        else:
            pad_positions = (attention_mask == 0).long()
            left_padding = pad_positions.cumprod(dim=-1).sum(dim=-1)
            first_valid_index = (attention_mask == 1).long().argmax(dim=-1, keepdim=True)
            first_tokens = input_sequences.gather(1, first_valid_index).squeeze(1)
            bos_offsets = (first_tokens == bos_token_identifier).long() if bos_token_identifier is not None else 0

        total_offsets = left_padding + bos_offsets

        if image_inputs.get("image_token_indices") is not None:
            shifted = []
            for mask, offset in zip(image_inputs["image_token_indices"], total_offsets):
                shifted.append(np.where(mask < 0, mask, mask + int(offset)))
            image_inputs["image_token_indices"] = shifted

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

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


__all__ = ["MolmoProcessor"]
