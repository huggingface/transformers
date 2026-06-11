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
from ...image_processing_backends import PilBackend, TorchvisionBackend
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging
from ..got_ocr2.configuration_got_ocr2 import GotOcr2Config


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="PaddlePaddle/PP-Chart2Table_safetensors")
@strict
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


@auto_docstring
class PPChart2TableImageProcessor(TorchvisionBackend):
    resample = 3
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 1024, "width": 1024}
    do_resize = True
    do_rescale = True
    do_normalize = True


@auto_docstring
class PPChart2TableImageProcessorPil(PilBackend):
    resample = 3
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    size = {"height": 1024, "width": 1024}
    do_resize = True
    do_rescale = True
    do_normalize = True


@auto_docstring
class PPChart2TableProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        if text is None or images is None:
            raise ValueError("Both `images` and `text` must be provided")
        return super().__call__(images=images, text=text, **kwargs)


__all__ = [
    "PPChart2TableConfig",
    "PPChart2TableImageProcessor",
    "PPChart2TableImageProcessorPil",
    "PPChart2TableProcessor",
]
