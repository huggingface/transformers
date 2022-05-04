# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import GPT2LMHeadModel


@require_torch
@require_sentencepiece
@require_tokenizers
class MegatronGPT2IntegrationTest(unittest.TestCase):
    @slow
    @unittest.skip("Model is not available.")
    def test_inference_no_head(self):
        directory = "nvidia/megatron-gpt2-345m/"
        if "MYDIR" in os.environ:
            directory = os.path.join(os.environ["MYDIR"], directory)
        model = GPT2LMHeadModel.from_pretrained(directory)
        model.to(torch_device)
        model.half()

        input_ids = torch.tensor(
            [[101, 7110, 1005, 1056, 2023, 11333, 17413, 1029, 102]],
            device=torch_device,
            dtype=torch.long,
        )

        with torch.no_grad():
            output = model(input_ids).logits

        expected_shape = torch.Size((1, 9, 50257))
        self.assertEqual(output.shape, expected_shape)

        expected_diag = torch.tensor(
            [
                4.9414,
                -0.2920,
                -1.2148,
                -4.0273,
                -0.5161,
                -5.2109,
                -1.2412,
                -1.8301,
                -1.7734,
                -4.7148,
                -0.2317,
                -1.0811,
                -2.1777,
                0.4141,
                -3.7969,
                -4.0586,
                -2.5332,
                -3.3809,
                4.3867,
            ],
            device=torch_device,
            dtype=torch.half,
        )

        for i in range(19):
            r, c = 8 * i // 17, 2792 * i  # along the diagonal
            computed, expected = output[0, r, c], expected_diag[i]
            msg = f"row={r} col={c} computed={computed} expected={expected}"
            self.assertAlmostEqual(computed, expected, delta=1e-4, msg=msg)
