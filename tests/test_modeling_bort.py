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

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device


if is_torch_available():
    import torch

    from transformers import AutoModel


@require_torch
@require_sentencepiece
@require_tokenizers
class BortIntegrationTest(unittest.TestCase):
    @slow
    def test_output_embeds_base_model(self):
        model = AutoModel.from_pretrained("amazon/bort")
        model.to(torch_device)

        input_ids = torch.tensor(
            [[0, 18077, 4082, 7804, 8606, 6195, 2457, 3321, 11, 10489, 16, 269, 2579, 328, 2]],
            device=torch_device,
            dtype=torch.long,
        )  # Schloß Nymphenburg in Munich is really nice!
        output = model(input_ids)["last_hidden_state"]
        expected_shape = torch.Size((1, 15, 1024))
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-0.0349, 0.0436, -1.8654], [-0.6964, 0.0835, -1.7393], [-0.9819, 0.2956, -0.2868]]],
            device=torch_device,
            dtype=torch.float,
        )
        self.assertTrue(torch.allclose(output[:, :3, :3], expected_slice, atol=1e-4))
