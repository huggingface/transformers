# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
Processor class for BridgeTower.
"""

from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring
from ...utils.deprecation import deprecate_attr


@auto_docstring
class BridgeTowerProcessor(ProcessorMixin):
    images_kwargs = {
        "do_normalize": True,
        "do_center_crop": True,
    }

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)


__getattr__ = deprecate_attr(__name__, rename={"BridgeTowerProcessorKwargs": (ProcessingKwargs, "5.29.0")})

__all__ = ["BridgeTowerProcessor"]
