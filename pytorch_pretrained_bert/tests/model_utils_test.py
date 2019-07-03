# coding=utf-8
# Copyright 2018 HuggingFace Inc..
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

from pytorch_pretrained_bert import PretrainedConfig, PreTrainedModel
from pytorch_pretrained_bert.modeling_bert import BertModel, BertConfig, PRETRAINED_MODEL_ARCHIVE_MAP, PRETRAINED_CONFIG_ARCHIVE_MAP


class ModelUtilsTest(unittest.TestCase):
    def test_model_from_pretrained(self):
        for model_name in list(PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = BertConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PretrainedConfig)

            model = BertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, PreTrainedModel)

            config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            self.assertEqual(model.config.output_attentions, True)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)

if __name__ == "__main__":
    unittest.main()
