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
"""Testing suite for the PyTorch GraniteMoeShared model."""

import unittest

from transformers import AutoTokenizer, GraniteMoeSharedConfig, is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        GraniteMoeSharedForCausalLM,
        GraniteMoeSharedModel,
    )


class GraniteMoeSharedModelTester:
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
        shared_intermediate_size=174,
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
        self.shared_intermediate_size = shared_intermediate_size
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
        return GraniteMoeSharedConfig(
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
            shared_intermediate_size=self.shared_intermediate_size,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = GraniteMoeSharedModel(config=config)
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
class GraniteMoeSharedModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            GraniteMoeSharedModel,
            GraniteMoeSharedForCausalLM,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "text-embedding": GraniteMoeSharedModel,
            "text-generation": GraniteMoeSharedForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    def setUp(self):
        self.model_tester = GraniteMoeSharedModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GraniteMoeSharedConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


@require_torch_accelerator
class GraniteMoeSharedIntegrationTest(unittest.TestCase):
    @slow
    def test_model_3b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = GraniteMoeSharedForCausalLM.from_pretrained("ibm/PowerMoE-3b", device_map="auto")

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        EXPECTED_MEANS = Expectations(
            {
                ("xpu", 3): torch.tensor([[-4.4005, -3.6689, -3.6187, -2.8308, -3.9871, -3.1001, -2.8738, -2.8063]]),
                ("cuda", 7): torch.tensor([[-2.2122, -1.6632, -2.9269, -2.3344, -2.0143, -3.0146, -2.6839, -2.5610]]),
                ("cuda", 8): torch.tensor([[-4.4005, -3.6689, -3.6187, -2.8308, -3.9871, -3.1001, -2.8738, -2.8063]]),
            }
        )

        EXPECTED_MEAN = EXPECTED_MEANS.get_expectation()
        torch.testing.assert_close(EXPECTED_MEAN.to(torch_device), out.logits.float().mean(-1), rtol=1e-2, atol=1e-2)

        # slicing logits[0, 0, 0:15]
        EXPECTED_SLICES = Expectations(
            {
                ("xpu", 3): torch.tensor([[2.5479, -9.2123, -9.2121, -9.2175, -9.2122, -1.5024, -9.2121, -9.2122, -9.2161, -9.2122, -6.3100, -3.6223, -3.6377, -5.2542, -5.2523]]),
                ("cuda", 7): torch.tensor([[4.8785, -2.2890, -2.2892, -2.2885, -2.2890, -3.5007, -2.2897, -2.2892, -2.2895, -2.2891, -2.2887, -2.2882, -2.2889, -2.2898, -2.2892]]),
                ("cuda", 8): torch.tensor([[2.5479, -9.2123, -9.2121, -9.2175, -9.2122, -1.5024, -9.2121, -9.2122, -9.2161, -9.2122, -6.3100, -3.6223, -3.6377, -5.2542, -5.2523]]),
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
    def test_model_3b_generation(self):
        # fmt: off
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("xpu", 3): (
                    "Simply put, the theory of relativity states that 1) the speed of light is constant, and 2) the speed of light is the same for all observers.\n\n"
                    "The first part is easy to understand. The second part is a little more difficult.\n\n"
                    "The second part of the theory of relativity is a little more difficult to understand.\n"
                ),
                ("cuda", 7): (
                    "Simply put, the theory of relativity states that \n$$\n\\frac{d^2x^\\mu}{d\\tau^2} = "
                    "\\frac{1}{c^2}\\frac{d^2x^\\mu}{dt^2}\n$$\nwhere $x^\\mu$ is a four-vector, $\\tau$ is the proper time"
                ),
                ("cuda", 8): (
                    "Simply put, the theory of relativity states that 1) the speed of light is constant, and 2) the speed of light is the same for all observers.\n\n"
                    "The first part is easy to understand. The second part is a little more difficult.\n\n"
                    "The second part of the theory of relativity is a little more difficult to understand.\n"
                ),
            }
        )
        # fmt: on
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()

        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("ibm/PowerMoE-3b")
        model = GraniteMoeSharedForCausalLM.from_pretrained("ibm/PowerMoE-3b", device_map="auto")
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(**model_inputs, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
