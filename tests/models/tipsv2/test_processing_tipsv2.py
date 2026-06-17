# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers import Tipsv2Processor
from transformers.testing_utils import require_sentencepiece, require_torch, require_vision

from ...test_processing_common import ProcessorTesterMixin


@require_sentencepiece
@require_torch
@require_vision
class Tipsv2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Tipsv2Processor
    model_id = "guarin/tipsv2-b14"  # TODO: switch to google repo
    image_text_kwargs_max_length = 64
    image_text_kwargs_override_max_length = 32
    image_unstructured_max_length = 48
