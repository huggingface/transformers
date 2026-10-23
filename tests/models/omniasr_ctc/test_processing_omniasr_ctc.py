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

from transformers import OmniASRCTCProcessor
from transformers.testing_utils import require_torch

from ...test_processing_common import ProcessorTesterMixin


@require_torch
class OmniASRCTCProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmniASRCTCProcessor
    audio_input_name = "input_values"
    text_input_name = "labels"
    model_id = "bezzam/omniasr-ctc-300m-v2"
