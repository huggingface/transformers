# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import unittest
import pytest
import torch

from pytorch_transformers.modeling_roberta import (RobertaForMaskedLM,
                                                   RobertaModel)


class RobertaModelTest(unittest.TestCase):

    # @pytest.mark.slow
    def test_inference_masked_lm(self):
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        
        input_ids = torch.tensor([[    0, 31414,   232,   328,   740,  1140, 12695,    69, 46078,  1588,   2]])
        output = model(input_ids)[0]
        expected_shape = torch.Size((1, 11, 50265))
        self.assertEqual(
            output.shape,
            expected_shape
        )
        # compare the actual values for a slice.
        expected_slice = torch.Tensor(
            [[[33.8843, -4.3107, 22.7779],
              [ 4.6533, -2.8099, 13.6252],
              [ 1.8222, -3.6898,  8.8600]]]
        )
        self.assertTrue(
            torch.allclose(output[:, :3, :3], expected_slice, atol=1e-3)
        )

    # @pytest.mark.slow
    def test_inference_no_head(self):
        model = RobertaModel.from_pretrained('roberta-base')
        
        input_ids = torch.tensor([[    0, 31414,   232,   328,   740,  1140, 12695,    69, 46078,  1588,   2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = torch.Tensor(
            [[[-0.0231,  0.0782,  0.0074],
              [-0.1854,  0.0539, -0.0174],
              [ 0.0548,  0.0799,  0.1687]]]
        )
        self.assertTrue(
            torch.allclose(output[:, :3, :3], expected_slice, atol=1e-3)
        )



if __name__ == '__main__':
    unittest.main()
