# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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
Processor class for MiniCPMV.
"""

import re
from typing import Optional, Union

import torch

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType
from .image_processing_minicpm import MiniCPMVBatchFeature


class MiniCPMVImageKwargs(ImagesKwargs, total=False):
    max_slice_nums: Optional[int]
    use_image_id: bool


class MiniCPM_V_4ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MiniCPMVImageKwargs
    _defaults = {
        "images_kwargs": {
            "do_pad": True,
            "max_slice_nums": None,
            "use_image_id": None,
        },
        "return_tensors": TensorType.PYTORCH,
    }


class MiniCPM_V_4Processor(ProcessorMixin):
    r"""
    Constructs a MiniCPMV processor which wraps a MiniCPMV image processor and a MiniCPMV tokenizer into a single processor.

    [`MiniCPMVProcessor`] offers all the functionalities of [`MiniCPMVImageProcessor`] and [`LlamaTokenizerWrapper`]. See the
    [`~MiniCPMVProcessor.__call__`] and [`~MiniCPMVProcessor.decode`] for more information.

    Args:
        image_processor ([`MiniCPMVImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerWrapper`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)
        self.version = image_processor.version

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]],
        images: ImageInput = None,
        **kwargs: Unpack[MiniCPM_V_4ProcessorKwargs],
    ) -> MiniCPMVBatchFeature:
        output_kwargs = self._merge_kwargs(
            MiniCPM_V_4ProcessorKwargs, self.tokenizer.init_kwargs, **kwargs
        )
        image_kwargs = output_kwargs["images_kwargs"]

        if images is not None:
            image_inputs = self.image_processor(images, **image_kwargs)
        return self._convert_images_texts_to_inputs(image_inputs, text, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        output_ids = args[0]
        result_text = []
        for result in output_ids:
            result = result[result != 0]
            result_text.append(self.tokenizer.decode(result, *args[1:], **kwargs).strip())
        return result_text
        # return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        result = args[0]
        result = result[result != 0]
        return self.tokenizer.decode(result, *args[1:], **kwargs).strip()

    def get_grid_placeholder(self, grid):
        if grid is None:
            return ""
        slice_image_placeholder = (
            self.tokenizer.slice_start
            + self.tokenizer.unk * self.image_processor.image_feature_size
            + self.tokenizer.slice_end
        )

        cols = grid[0]
        rows = grid[1]
        slices = []
        for i in range(rows):
            lines = []
            for j in range(cols):
                lines.append(slice_image_placeholder)
            slices.append("".join(lines))

        slice_placeholder = "\n".join(slices)
        return slice_placeholder

    def get_image_id_placeholder(self, idx=0):
        return f"{self.tokenizer.im_id_start}{idx}{self.tokenizer.im_id_end}"

    def get_slice_image_placeholder(self, image_size, image_idx=0, max_slice_nums=None, use_image_id=None):
        max_slice_nums = self.image_processor.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        assert max_slice_nums > 0
        grid = self.image_processor.get_sliced_grid(image_size=image_size, max_slice_nums=max_slice_nums)

        image_placeholder = (
            self.tokenizer.im_start
            + self.tokenizer.unk * self.image_processor.image_feature_size
            + self.tokenizer.im_end
        )
        use_image_id = self.image_processor.use_image_id if use_image_id is None else bool(use_image_id)
        if use_image_id:
            final_placeholder = self.get_image_id_placeholder(image_idx) + image_placeholder
        else:
            final_placeholder = image_placeholder

        if self.image_processor.slice_mode:
            final_placeholder = final_placeholder + self.get_grid_placeholder(grid=grid)
        return final_placeholder

    def _convert(
        self, input_str, max_inp_length: Optional[int] = None
    ):
        input_ids = self.tokenizer.encode(input_str)
        if max_inp_length is not None:
            input_ids = input_ids[:max_inp_length]
        input_ids = torch.tensor(input_ids, dtype=torch.int32)

        start_cond = (input_ids == self.tokenizer.im_start_id) | (input_ids == self.tokenizer.slice_start_id)
        end_cond = (input_ids == self.tokenizer.im_end_id) | (input_ids == self.tokenizer.slice_end_id)

        image_start_tokens = torch.where(start_cond)[0]
        image_start_tokens += 1
        image_end_tokens = torch.where(end_cond)[0]

        valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))

        image_bounds = torch.hstack(
            [
                image_start_tokens[:valid_image_nums].unsqueeze(-1),
                image_end_tokens[:valid_image_nums].unsqueeze(-1),
            ]
        )
        return input_ids, image_bounds

    def _convert_images_texts_to_inputs(
            self,
            images,
            texts: Union[str, list[str]],
            truncation=None,
            max_length=None,
            max_slice_nums=None,
            use_image_id=None,
            return_tensors=None,
            **kwargs
        ):
        if images is None or not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, truncation=truncation, max_length=max_length, **kwargs)
            return MiniCPMVBatchFeature(data={**model_inputs})

        pattern = "(<image>./</image>)"
        images, image_sizes, tgt_sizes = images["pixel_values"], images["image_sizes"], images["tgt_sizes"]

        if isinstance(texts, str):
            texts = [texts]
        input_ids_list = []
        image_bounds_list = []
        for index, text in enumerate(texts):
            image_tags = re.findall(pattern, text)
            assert len(image_tags) == len(image_sizes[index])
            text_chunks = text.split(pattern)
            final_text = ""
            for i in range(len(image_tags)):
                final_text = final_text + text_chunks[i] + \
                    self.get_slice_image_placeholder(
                        image_sizes[index][i],
                        i,
                        max_slice_nums,
                        use_image_id
                    )
            final_text += text_chunks[-1]
            input_ids, image_bounds = self._convert(final_text, max_length)
            input_ids_list.append(input_ids)
            image_bounds_list.append(image_bounds)
        padded_input_ids, padding_lengths = self.pad(
            input_ids_list,
            padding_side="left"
        )
        attention_mask = torch.ones_like(padded_input_ids, dtype=torch.bool)
        for i, length in enumerate(padding_lengths):
            image_bounds_list[i] = image_bounds_list[i] + length
            attention_mask[i, :length] = False

        return MiniCPMVBatchFeature(data={
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "pixel_values": images,
            "image_sizes": image_sizes,
            "image_bound": image_bounds_list,
            "tgt_sizes": tgt_sizes
        })

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


    def pad(self, inputs, max_length=None, padding_value=0, padding_side="left"):
        items = []
        if isinstance(inputs[0], list):
            assert isinstance(inputs[0][0], torch.Tensor)
            for it in inputs:
                for tr in it:
                    items.append(tr)
        else:
            assert isinstance(inputs[0], torch.Tensor)
            items = inputs

        batch_size = len(items)
        shape = items[0].shape
        dim = len(shape)
        assert dim <= 2
        if max_length is None:
            max_length = 0
        max_length = max(max_length, max(item.shape[-1] for item in items))
        min_length = min(item.shape[-1] for item in items)
        dtype = items[0].dtype

        if dim == 0:
            return torch.stack(list(items), dim=0), [0]
        elif dim == 1:
            if max_length == min_length:
                return torch.stack(list(items), dim=0), [0] * batch_size
            tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
        else:
            tensor = (
                torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
                + padding_value
            )

        padding_length = []
        for i, item in enumerate(items):
            if dim == 1:
                if padding_side == "left":
                    tensor[i, -len(item) :] = item.clone()
                else:
                    tensor[i, : len(item)] = item.clone()
            elif dim == 2:
                if padding_side == "left":
                    tensor[i, -len(item) :, :] = item.clone()
                else:
                    tensor[i, : len(item), :] = item.clone()
            padding_length.append(tensor.shape[-1] - len(item))

        return tensor, padding_length


__all__ = ["MiniCPM_V_4Processor"]
