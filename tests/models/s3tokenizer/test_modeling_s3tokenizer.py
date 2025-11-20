# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch S3Tokenizer model."""

import unittest

import numpy as np

from transformers import S3TokenizerConfig
from transformers.testing_utils import is_torch_available, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch

    from transformers import S3TokenizerModel


@require_torch
class S3TokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=400,
        is_training=False,
        use_labels=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        return S3TokenizerConfig(
            n_mels=80,
            n_audio_state=512,
            n_audio_head=8,
            n_audio_layer=6,
            vocab_size=6561,
            n_fft=400,
            hop_length=160,
            sampling_rate=16000,
            use_sdpa=False,
        )

    def create_and_check_model(self, config, input_values):
        model = S3TokenizerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_values)
        self.parent.assertIsNotNone(result.speech_tokens)
        self.parent.assertIsNotNone(result.speech_token_lens)


@require_torch
class S3TokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (S3TokenizerModel,) if is_torch_available() else ()
    is_encoder_decoder = False
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = S3TokenizerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=S3TokenizerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, inputs_dict["input_values"])

    @unittest.skip(reason="S3Tokenizer does not output hidden states in the traditional sense")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not have attention weights in the traditional sense")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support input embeddings")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support training mode")
    def test_training(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support training mode")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not support retain_grad")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not have typical model forward signature")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="S3Tokenizer model does not support typical model features")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="S3Tokenizer does not use feed forward chunking")
    def test_feed_forward_chunking(self):
        pass

    @slow
    @require_torch
    def test_model_from_pretrained(self):
        pass
