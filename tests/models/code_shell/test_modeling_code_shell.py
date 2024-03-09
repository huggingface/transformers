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
""" Testing suite for the PyTorch CodeShell model. """


import unittest

from ...test_modeling_common import floats_tensor
from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from transformers import CodeShellConfig
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        CodeShellModel,
        CodeShellForCausalLM,
    )


class CodeShellModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        embed_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        rope_scaling=None,
        pad_token_id=70000,
        position_embedding_type="rope",
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.embed_dropout_prob = embed_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.rope_scaling = rope_scaling
        self.position_embedding_type = position_embedding_type
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, None, input_mask

    def get_config(self):
        return CodeShellConfig(
            vocab_size=self.vocab_size,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            n_inner=self.intermediate_size,
            activation_function=self.hidden_act,
            resid_pdrop=self.hidden_dropout_prob,
            embd_pdrop=self.embed_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            n_positions=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            pad_token_id=self.pad_token_id,
            position_embedding_type=self.position_embedding_type,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(
            self, config, input_ids, token_type_ids, input_mask
    ):
        model = CodeShellModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
            self,
            config,
            input_ids,
            token_type_ids,
            input_mask,
            encoder_hidden_states,
            encoder_attention_mask,
    ):
        model = CodeShellForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class CodeShellModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            CodeShellModel,
            CodeShellForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (CodeShellForCausalLM,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = CodeShellModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CodeShellConfig, n_embd=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

@require_torch
class CodeShellModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_causal_lm(self):
        # model = CodeShellForCausalLM.from_pretrained("WisdomShell/CodeShell-7B")
        model = CodeShellForCausalLM.from_pretrained("/shd/eval/models/CodeShell-7B-20240104")
        input_ids = torch.tensor([[[13723,  1086]]])
        output = model(input_ids)[0]

        # TODO Replace vocab size
        vocab_size = 70000

        expected_shape = torch.Size((1, 2, vocab_size))
        self.assertEqual(output.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        expected_slice = torch.tensor(
            [[[-4.0345, -5.6830], [-2.7156, -4.2179]]]
        )

        self.assertTrue(torch.allclose(output[:, :2, :2], expected_slice, atol=1e-4))
