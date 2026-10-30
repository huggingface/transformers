# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Kosmos2_5.
"""

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import TextInput
from ...utils import auto_docstring


class Kosmos2_5ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
            "stride": 0,
            "truncation": True,
        },
        "images_kwargs": {
            "max_patches": 4096,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@auto_docstring
class Kosmos2_5Processor(ProcessorMixin):
    valid_processor_kwargs = Kosmos2_5ProcessorKwargs

    def __init__(self, image_processor, tokenizer, num_image_tokens: int = 2048):
        r"""
        num_image_tokens (`int`, *optional*, defaults to 2048):
            Number of image tokens used as a placeholder.
        """
        self.image_start_token = tokenizer.boi_token  # "<image>" : fixed token for the start of image
        self.image_end_token = tokenizer.eoi_token  # "</image>" : fixed token for the end of image
        self.image_token = tokenizer.image_token  # "<s>" : within a <image> ... </image> pair, these <s> tokens indicate they are positions reserved for an image
        self.num_image_tokens = num_image_tokens
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] = None,
        **kwargs: Unpack[Kosmos2_5ProcessorKwargs],
    ) -> BatchFeature:
        model_inputs = super().__call__(images=images, text=text, **kwargs)
        model_inputs["image_embeds_position_mask"] = model_inputs.pop("mm_token_type_ids", None)
        return model_inputs

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | list[TextInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, *_ = super().prepare_inputs_layout(text=text, images=images)
        if images is not None:
            prepend_prompt = (
                f"{self.tokenizer.bos_token}{self.image_start_token}{self.image_token}{self.image_end_token}"
            )
            text = [f"{prepend_prompt}{sample}" for sample in text]
        return images, text, None, None

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        return self.image_token * self.num_image_tokens

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Kosmos2_5TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    @property
    def unused_input_names(self) -> list[str]:
        return ["rows", "cols"]


__all__ = ["Kosmos2_5Processor"]
