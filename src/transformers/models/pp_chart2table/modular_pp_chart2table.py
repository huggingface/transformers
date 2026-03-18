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

from huggingface_hub.dataclasses import strict

from ...feature_extraction_utils import BatchFeature
from ...image_processing_utils_fast import BaseImageProcessorFast
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ...utils.import_utils import requires
from ..got_ocr2.configuration_got_ocr2 import GotOcr2Config


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-Chart2Table_safetensors")
@strict(accept_kwargs=True)
class PPChart2TableConfig(GotOcr2Config):
    model_type = "pp_chart2table"

    r"""
    Example:

    ```python
    >>> from transformers import GotOcr2ForConditionalGeneration, PPChart2TableConfig

    >>> # Initializing a PPChart2Table style configuration
    >>> configuration = PPChart2TableConfig()

    >>> # Initializing a model from the PaddlePaddle/PP-Chart2Table_safetensors style configuration
    >>> model = GotOcr2ForConditionalGeneration(configuration)  # underlying architecture is Got Ocr 2

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""


class PPChart2TableImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    patch_size (`int`, *optional*, defaults to `16`):
        The expected patch size out of the image processor.
    num_patches (`int`, *optional*, defaults to `16`):
        Alias for `patch_size`.
    """

    patch_size: int
    num_patches: int


@auto_docstring
class PPChart2TableImageProcessorFast(BaseImageProcessorFast):
    resample = 3
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 1024, "width": 1024}
    patch_size = 16
    num_patches = 16
    do_resize = True
    do_rescale = True
    do_normalize = True
    valid_kwargs = PPChart2TableImageProcessorKwargs


@auto_docstring
@requires(backends=("torch",))
class PPChart2TableProcessor(ProcessorMixin):
    model_input_names = ["input_ids", "pixel_values"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        # PPChart2TableProcessor uses hardcoded "Chart to table" instruction internally via chat template
        self.messages = [
            {
                "role": "system",
            },
            {
                "role": "user",
                "image": {"num_patches": self.image_processor.num_patches},
            },
        ]

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            ProcessingKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is None:
            raise ValueError("At least one of `images` must be provided")
        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        batch_size = image_inputs["pixel_values"].shape[0]

        # Use tokenizer's apply_chat_template instead of manually loading template
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            **output_kwargs["text_kwargs"],
        )

        # Prepare input ids for batch
        input_ids = inputs["input_ids"].repeat(batch_size, 1)

        return BatchFeature(data={"input_ids": input_ids, **image_inputs})


__all__ = [
    "PPChart2TableConfig",
    "PPChart2TableImageProcessorFast",
    "PPChart2TableProcessor",
]
