# coding=utf-8
# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor
from .utils import CACHE_DIR, require_torch, slow, torch_device


if is_torch_available():
    import torch
    from transformers import BertConfig, BertModel, MarianModel, MarianConfig


class ModelTester:
    def __init__(
        self, parent,
    ):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_labels = False
        self.vocab_size = 99
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 4
        # self.hidden_act = "gelu"
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 20
        self.eos_token_id = 2
        self.pad_token_id = 1
        self.bos_token_id = 0
        torch.manual_seed(0)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(3,)
        input_ids[:, -1] = 2  # Eos Token
        config = MarianConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            # decoder_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            # decoder_ffn_dim=self.intermediate_size,
            # dropout=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

        inputs_dict = {"input_ids": input_ids, "decoder_input_ids": input_ids[:, 3:]}
        return config, inputs_dict


@require_torch
class Bert2BertModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (MarianModel,) if is_torch_available() else ()
    all_generative_model_classes = all_model_classes
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_resize_embeddings = False
    is_encoder_decoder = True


    def setUp(self):
        self.model_tester = ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BertConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip("Passing inputs_embeds not implemented for Bart.")
    def test_inputs_embeds(self):
        pass
