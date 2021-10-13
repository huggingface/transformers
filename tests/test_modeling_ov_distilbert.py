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

from transformers import AlbertConfig, is_ov_available
from transformers.testing_utils import require_ov, slow

if is_ov_available():
    from transformers import OVAutoModel

@require_ov
class OVDistilBertModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", from_tf=True)
        input_ids = np.array([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = (1, 6, 768)
        self.assertEqual(output.shape, expected_shape)

        expected_slice = np.array(
            [
                [
                    [0.19261885, -0.13732955, 0.4119799],
                    [0.22150156, -0.07422661, 0.39037204],
                    [0.22756018, -0.0896414, 0.3701467],
                ]
            ]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
