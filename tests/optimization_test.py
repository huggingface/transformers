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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import torch

from pytorch_pretrained_bert import BertAdam

class OptimizationTest(unittest.TestCase):

    def assertListAlmostEqual(self, list1, list2, tol):
        self.assertEqual(len(list1), len(list2))
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, delta=tol)

    def test_adam(self):
        w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
        target = torch.tensor([0.4, 0.2, -0.5])
        criterion = torch.nn.MSELoss()
        # No warmup, constant schedule, no gradient clipping
        optimizer = BertAdam(params=[w], lr=2e-1,
                                          weight_decay=0.0,
                                          max_grad_norm=-1)
        for _ in range(100):
            loss = criterion(w, target)
            loss.backward()
            optimizer.step()
            w.grad.detach_() # No zero_grad() function on simple tensors. we do it ourselves.
            w.grad.zero_()
        self.assertListAlmostEqual(w.tolist(), [0.4, 0.2, -0.5], tol=1e-2)


if __name__ == "__main__":
    unittest.main()
