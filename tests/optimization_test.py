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
from pytorch_pretrained_bert import OpenAIAdam
from pytorch_pretrained_bert.optimization import ConstantLR, WarmupLinearSchedule, WarmupConstantSchedule, \
    WarmupCosineWithWarmupRestartsSchedule, WarmupCosineWithHardRestartsSchedule, WarmupCosineSchedule
import numpy as np


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


class ScheduleInitTest(unittest.TestCase):
    def test_bert_sched_init(self):
        m = torch.nn.Linear(50, 50)
        optim = BertAdam(m.parameters(), lr=0.001, warmup=.1, t_total=1000, schedule=None)
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], ConstantLR))
        optim = BertAdam(m.parameters(), lr=0.001, warmup=.1, t_total=1000, schedule="none")
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], ConstantLR))
        optim = BertAdam(m.parameters(), lr=0.001, warmup=.01, t_total=1000)
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], WarmupLinearSchedule))
        # shouldn't fail

    def test_openai_sched_init(self):
        m = torch.nn.Linear(50, 50)
        optim = OpenAIAdam(m.parameters(), lr=0.001, warmup=.1, t_total=1000, schedule=None)
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], ConstantLR))
        optim = OpenAIAdam(m.parameters(), lr=0.001, warmup=.1, t_total=1000, schedule="none")
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], ConstantLR))
        optim = OpenAIAdam(m.parameters(), lr=0.001, warmup=.01, t_total=1000)
        self.assertTrue(isinstance(optim.param_groups[0]["schedule"], WarmupLinearSchedule))
        # shouldn't fail


class WarmupCosineWithRestartsTest(unittest.TestCase):
    def test_it(self):
        m = WarmupCosineWithWarmupRestartsSchedule(warmup=0.05, t_total=1000., cycles=5)
        x = np.arange(0, 1000)
        y = [m.get_lr(xe) for xe in x]
        y = np.asarray(y)
        expected_zeros = y[[0, 200, 400, 600, 800]]
        print(expected_zeros)
        expected_ones = y[[50, 250, 450, 650, 850]]
        print(expected_ones)
        self.assertTrue(np.allclose(expected_ones, 1))
        self.assertTrue(np.allclose(expected_zeros, 0))


if __name__ == "__main__":
    unittest.main()
