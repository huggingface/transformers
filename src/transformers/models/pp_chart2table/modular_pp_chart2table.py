# Copyright 2026 The PaddlePaddle Team and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ..got_ocr2.configuration_got_ocr2 import GotOcr2Config
from ..got_ocr2.modeling_got_ocr2 import (
    GotOcr2ForConditionalGeneration,
    GotOcr2Model,
    GotOcr2ModelOutputWithPast,
    GotOcr2PreTrainedModel,
    GotOcr2VisionEncoder,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-Chart2Table_safetensors")
class PPChart2TableConfig(GotOcr2Config):
    pass


@auto_docstring
class PPChart2TableImageProcessorFast(BaseImageProcessorFast):
    resample = 3
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 1024, "width": 1024}
    patch_size = 16
    merge_size = 4
    do_resize = True
    do_rescale = True
    do_normalize = True


@auto_docstring
class PPChart2TableProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pt")
        else:
            image_inputs = {}

        batch_size, _, height, _ = image_inputs["pixel_values"].shape
        num_patches = height // self.image_processor.patch_size // self.image_processor.merge_size

        messages = [
            {
                "role": "system",
            },
            {
                "role": "user",
                "image": {"num_patches": num_patches},
            },
        ]

        # Use tokenizer's apply_chat_template instead of manually loading template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Prepare input ids for batch
        input_ids = inputs["input_ids"].repeat(batch_size, 1)

        return BatchFeature(data={"input_ids": input_ids, **image_inputs})


class PPChart2TableVisionPreTrainedModel(GotOcr2PreTrainedModel):
    pass


class PPChart2TableVisionEncoder(GotOcr2VisionEncoder, PPChart2TableVisionPreTrainedModel):
    pass


@dataclass
class PPChart2TableModelOutputWithPast(GotOcr2ModelOutputWithPast):
    pass


@auto_docstring
class PPChart2TablePreTrainedModel(GotOcr2PreTrainedModel):
    pass


@auto_docstring
class PPChart2TableModel(GotOcr2Model):
    pass


@auto_docstring(
    custom_intro="""
    PP-Chart2Table model for conditional generation (table text generation from chart images),
    extending the core model with a language modeling (LM) head and generation utilities.
    """
)
class PPChart2TableForConditionalGeneration(GotOcr2ForConditionalGeneration):
    pass


__all__ = [
    "PPChart2TableForConditionalGeneration",
    "PPChart2TableModel",
    "PPChart2TableConfig",
    "PPChart2TableVisionPreTrainedModel",
    "PPChart2TablePreTrainedModel",
    "PPChart2TableImageProcessorFast",
    "PPChart2TableProcessor",
]
