# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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

from transformers import HYV4FlashProcessor
from transformers.testing_utils import require_tokenizers, require_torchvision

from ..hy_v4_vl import test_processing_hy_v4_vl


@require_torchvision
@require_tokenizers
class HYV4FlashProcessorTest(test_processing_hy_v4_vl.HYV4VLProcessorTest):
    processor_class = HYV4FlashProcessor
