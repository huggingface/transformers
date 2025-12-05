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
"""Testing suite for the PyTorch Granite model."""

import unittest

from transformers import GraniteConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        GraniteForCausalLM,
        GraniteModel,
    )


class GraniteModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return GraniteConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GraniteModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class GraniteModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GraniteModel,
            GraniteForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": GraniteModel,
            "text-generation": GraniteForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = GraniteModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


@require_torch_accelerator
class GraniteIntegrationTest(unittest.TestCase):
    @slow
    @require_read_token
    def test_model_3b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteForCausalLM.from_pretrained(
            "ibm/PowerLM-3b", device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-3.1406, -2.5469, -2.6250, -2.1250, -2.6250, -2.6562, -2.6875, -2.9688]]),
                ("cuda", 7): torch.tensor([[-1.9798, -3.1626, -2.8062, -2.3777, -2.7091, -2.2338, -2.5924, -2.3974]]),
                ("cuda", 8): torch.tensor([[-3.1406, -2.5469, -2.6250, -2.1250, -2.6250, -2.6562, -2.6875, -2.9688]]),
            }
        )
        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.mean(-1).float(), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICES = Expectations(
            {
                ("xpu", 3): torch.tensor([[2.2031, -5.0625, -5.0625, -5.0625, -5.0625, -0.9180, -5.0625, -5.0625, -5.0625, -5.0625, -5.5312, -2.1719, -1.7891, -0.4922, -2.5469]]),
                ("cuda", 7): torch.tensor([[4.8750, -2.1875, -2.1875, -2.1875, -2.1875, -2.8438, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875, -2.1875]]),
                ("cuda", 8): torch.tensor([[2.0938, -5.0312, -5.0312, -5.0312, -5.0312, -1.0469, -5.0312, -5.0312, -5.0312, -5.0312, -5.5625, -2.1875, -1.7891, -0.5820, -2.6250]]),
            }
        )
        EXPECTED_SLICE = EXPECTED_SLICES.get_expectation()
        # fmt: on
        self.assertTrue(
            torch.allclose(
                EXPECTED_SLICE.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    @slow
    @require_read_token
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteForCausalLM.from_pretrained("ibm/PowerLM-3b", device_map="auto", dtype=torch.float16)

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-3.2693, -2.5957, -2.6234, -2.1675, -2.6386, -2.6850, -2.7039, -2.9656]]),
                ("cuda", 7): torch.tensor([[-2.0984, -3.1294, -2.8153, -2.3568, -2.7337, -2.2624, -2.6016, -2.4022]]),
                ("cuda", 8): torch.tensor([[-3.2934, -2.6019, -2.6258, -2.1691, -2.6394, -2.6876, -2.7032, -2.9688]]),
            }
        )
        # fmt: on
        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()

        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)
