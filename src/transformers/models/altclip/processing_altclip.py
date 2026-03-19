# Copyright 2022 WenXiang ZhongzhiCheng LedellWu LiuGuang BoWenZhang The HuggingFace Inc. team. All rights reserved.
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
Image/Text processor class for AltCLIP
"""

from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring


@auto_docstring
class AltCLIPProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None):
        super().__init__(image_processor, tokenizer)


__all__ = ["AltCLIPProcessor"]
