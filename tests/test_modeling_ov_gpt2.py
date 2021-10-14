# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import numpy as np

from transformers import is_ov_available, is_torch_available
from transformers.testing_utils import require_ov, require_torch, slow

if is_ov_available():
    from transformers import OVAutoModel, AutoModel

if is_torch_available():
    from transformers import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


@require_ov
@require_torch
class GPT2ModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        for model_name in GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OVAutoModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)

            input_ids = np.random.randint(0, 255, (1, 6))
            attention_mask = np.random.randint(0, 2, (1, 6))

            expected_shape = (1, 6, 768)
            output = model(input_ids, attention_mask=attention_mask)[0]
            self.assertEqual(output.shape, expected_shape)
