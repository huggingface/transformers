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

    from transformers import UniTEForSequenceClassification


@require_sentencepiece
@require_tokenizers
@require_torch
class UniTEForSequenceClassificationIntegrationTest(unittest.TestCase):
    @slow
    def test_unite_up(self):
        model = UniTEForSequenceClassification.from_pretrained("ywan/unite-up")
        src_input = {'input_ids': torch.tensor([[0, 6, 124084, 38, 2, 1, 1],
                                                [0, 6, 222473, 41380, 994, 38, 2]])}
        hyp_input = {'input_ids': torch.tensor([[0, 2673, 38, 2, 1, 1, 1],
                                                [0, 73398, 47, 1957, 398, 38, 2]])}
        ref_input = {'input_ids': torch.tensor([[0, 35378, 38, 2, 1, 1, 1],
                                                [0, 73398, 47, 23356, 398, 38, 2]])}

        # src: ['你好！', '很高兴认识你！']
        # hyp: ['Hi!', 'Nice to see you!']
        # ref: ['Hello!', 'Nice to meet you!']

        expected_output_shape = torch.Size(2)  # batch_size, sequence_length, embedding_vector_dim
        src_only_expected_output_values_last_dim = torch.tensor([0.4785, 0.5464])
        ref_only_expected_output_values_last_dim = torch.tensor([0.7119, 0.7508])
        src_ref_expected_output_values_last_dim = torch.tensor([0.6266, 0.7369])

        #  xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')
        #  xlmr.eval()
        #  expected_output_values_last_dim = xlmr.extract_features(input_ids[0])[:, :, -1]

        src_only_output = model(hyp=hyp_input, src=src_input)
        ref_only_output = model(hyp=hyp_input, ref=ref_input)
        src_ref_output = model(hyp=hyp_input, src=src_input, ref=ref_input)

        self.assertEqual(src_only_output.shape, expected_output_shape)
        self.assertEqual(ref_only_output.shape, expected_output_shape)
        self.assertEqual(src_ref_output.shape, expected_output_shape)
        
        self.assertTrue(torch.allclose(src_only_output, src_only_expected_output_values_last_dim, atol=1e-3))
        self.assertTrue(torch.allclose(ref_only_output, ref_only_expected_output_values_last_dim, atol=1e-3))
        self.assertTrue(torch.allclose(src_ref_output, src_ref_expected_output_values_last_dim, atol=1e-3))

    @slow
    def test_unite_mup(self):
        model = UniTEForSequenceClassification.from_pretrained("ywan/unite-mup")
        src_input = {'input_ids': torch.tensor([[0, 6, 124084, 38, 2, 1, 1],
                                                [0, 6, 222473, 41380, 994, 38, 2]])}
        hyp_input = {'input_ids': torch.tensor([[0, 2673, 38, 2, 1, 1, 1],
                                                [0, 73398, 47, 1957, 398, 38, 2]])}
        ref_input = {'input_ids': torch.tensor([[0, 35378, 38, 2, 1, 1, 1],
                                                [0, 73398, 47, 23356, 398, 38, 2]])}

        # src: ['你好！', '很高兴认识你！']
        # hyp: ['Hi!', 'Nice to see you!']
        # ref: ['Hello!', 'Nice to meet you!']

        expected_output_shape = torch.Size(2)  # batch_size, sequence_length, embedding_vector_dim
        src_only_expected_output_values_last_dim = torch.tensor([0.7145, 0.6583])
        ref_only_expected_output_values_last_dim = torch.tensor([0.7465, 0.7588])
        src_ref_expected_output_values_last_dim = torch.tensor([0.6857, 0.7173])

        #  xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.base')
        #  xlmr.eval()
        #  expected_output_values_last_dim = xlmr.extract_features(input_ids[0])[:, :, -1]

        src_only_output = model(hyp=hyp_input, src=src_input)
        ref_only_output = model(hyp=hyp_input, ref=ref_input)
        src_ref_output = model(hyp=hyp_input, src=src_input, ref=ref_input)

        self.assertEqual(src_only_output.shape, expected_output_shape)
        self.assertEqual(ref_only_output.shape, expected_output_shape)
        self.assertEqual(src_ref_output.shape, expected_output_shape)
        
        self.assertTrue(torch.allclose(src_only_output, src_only_expected_output_values_last_dim, atol=1e-3))
        self.assertTrue(torch.allclose(ref_only_output, ref_only_expected_output_values_last_dim, atol=1e-3))
        self.assertTrue(torch.allclose(src_ref_output, src_ref_expected_output_values_last_dim, atol=1e-3))
