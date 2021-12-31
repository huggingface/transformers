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
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow


if is_torch_available():
    import torch

    from transformers import XLMRobertaXLModel


@require_torch
class XLMRobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_xlm_roberta_xlarge(self):
        model = XLMRobertaXLModel.from_pretrained("Soonhwan-Kwon/xlm-roberta-xlarge")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = torch.Size((1, 12, 2560))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[0.0110, 0.0605, 0.0354, 0.0689, 0.0066, 0.0691, 0.0302, 0.0412, 0.0860, 0.0036, 0.0405, 0.0170]]
        )

        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))

    @slow
    def test_xlm_roberta_xxlarge(self):
        model = XLMRobertaXLModel.from_pretrained("Soonhwan-Kwon/xlm-roberta-xxlarge")
        input_ids = torch.tensor([[0, 581, 10269, 83, 99942, 136, 60742, 23, 70, 80583, 18276, 2]])
        # The dog is cute and lives in the garden house

        expected_output_shape = torch.Size((1, 12, 4096))  # batch_size, sequence_length, embedding_vector_dim
        expected_output_values_last_dim = torch.tensor(
            [[0.0046, 0.0146, 0.0227, 0.0126, 0.0219, 0.0175, -0.0101, 0.0006, 0.0124, 0.0209, -0.0063, 0.0096]]
        )

        output = model(input_ids)["last_hidden_state"].detach()
        self.assertEqual(output.shape, expected_output_shape)
        # compare the actual values for a slice of last dim
        self.assertTrue(torch.allclose(output[:, :, -1], expected_output_values_last_dim, atol=1e-3))
