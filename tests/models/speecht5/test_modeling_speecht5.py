# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch SpeechT5 model. """

import math
import unittest

from transformers import SpeechT5Config, SpeechT5HiFiGANConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


# import numpy as np
# from datasets import load_dataset


if is_torch_available():
    from transformers import (
        SpeechT5ForCTC,
        SpeechT5ForSpeechToText,
        SpeechT5ForTextToSpeech,
        SpeechT5HiFiGAN,
        SpeechT5Model,
    )


class SpeechT5ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return SpeechT5Config(
            hidden_size=self.hidden_size,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = SpeechT5Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class SpeechT5ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (SpeechT5ForSpeechToText, SpeechT5ForCTC, SpeechT5ForTextToSpeech, SpeechT5Model)
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = SpeechT5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()


class SpeechT5HiFiGANModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return SpeechT5HiFiGANConfig(
            hidden_size=self.hidden_size,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = SpeechT5HiFiGANConfig(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class SpeechT5HiFiGANModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SpeechT5HiFiGAN) if is_torch_available() else ()
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = SpeechT5HiFiGANModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SpeechT5HiFiGANConfig)

    def test_config(self):
        self.config_tester.run_common_tests()
