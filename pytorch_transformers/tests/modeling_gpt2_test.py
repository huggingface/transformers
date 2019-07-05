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

import os
import unittest
import json
import random
import shutil
import pytest

import torch

from pytorch_transformers import (GPT2Config, GPT2Model,
                                     GPT2LMHeadModel, GPT2DoubleHeadsModel)

from .model_tests_commons import (create_and_check_commons, ConfigTester, GPTModelTester)

class GPT2ModelTest(unittest.TestCase):

    def test_config(self):
        config_tester = ConfigTester(self, config_class=GPT2Config, n_embd=37)
        config_tester.run_common_tests()

    def test_model(self):
        model_tester = GPTModelTester(self, config_class=GPT2Config, base_model_class=GPT2Model,
                                            lm_head_model_class=GPT2LMHeadModel,
                                            double_head_model_class=GPT2DoubleHeadsModel)
        model_tester.run_common_tests(test_presents=True)

    @pytest.mark.slow
    def test_pretrained(self):
        model_tester = GPTModelTester(self, config_class=GPT2Config, base_model_class=GPT2Model,
                                            lm_head_model_class=GPT2LMHeadModel,
                                            double_head_model_class=GPT2DoubleHeadsModel)
        model_tester.run_slow_tests()

if __name__ == "__main__":
    unittest.main()
