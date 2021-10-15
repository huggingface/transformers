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

from transformers import is_ov_available
from transformers.testing_utils import require_ov, require_torch, slow

if is_ov_available():
    from transformers import OVAutoModelWithLMHead, AutoModelWithLMHead


@require_ov
@require_torch
class OVOPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_openai_gpt(self):
        model = OVAutoModelWithLMHead.from_pretrained("openai-gpt", from_pt=True)
        input_ids = np.array([[481, 4735, 544]], dtype=np.int64)  # the president is
        expected_output_ids = [
            481,
            4735,
            544,
            246,
            963,
            870,
            762,
            239,
            244,
            40477,
            244,
            249,
            719,
            881,
            487,
            544,
            240,
            244,
            603,
            481,
        ]  # the president is a very good man. " \n " i\'m sure he is, " said the

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
