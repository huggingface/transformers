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
""" Testing suite for the PyTorch CpmBee model. """


import unittest

from transformers.testing_utils import is_torch_available, require_torch, tooslow

from ...generation.test_utils import torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        CpmBeeConfig,
        CpmBeeForCausalLM,
        CpmBeeModel,
        CpmBeeTokenizer,
    )


@require_torch
class CpmBeeModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        is_training=True,
        use_token_type_ids=False,
        use_input_mask=False,
        use_labels=False,
        use_mc_token_ids=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=37,
        num_buckets=32,
        max_distance=128,
        position_bias_num_segment_buckets=32,
        init_std=1.0,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_token_type_ids = use_token_type_ids
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.use_mc_token_ids = use_mc_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.position_bias_num_segment_buckets = position_bias_num_segment_buckets
        self.init_std = init_std
        self.return_dict = return_dict

    def prepare_config_and_inputs(self):
        input_ids = {}
        input_ids["input_ids"] = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).type(torch.int32)
        input_ids["use_cache"] = False

        config = self.get_config()

        return (config, input_ids)

    def get_config(self):
        return CpmBeeConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            dim_ff=self.intermediate_size,
            position_bias_num_buckets=self.num_buckets,
            position_bias_max_distance=self.max_distance,
            position_bias_num_segment_buckets=self.position_bias_num_segment_buckets,
            use_cache=True,
            init_std=self.init_std,
            return_dict=self.return_dict,
        )

    def create_and_check_cpmbee_model(self, config, input_ids, *args):
        model = CpmBeeModel(config=config)
        model.to(torch_device)
        model.eval()

        hidden_states = model(**input_ids).last_hidden_state

        self.parent.assertEqual(hidden_states.shape, (self.batch_size, self.seq_length, config.hidden_size))

    def create_and_check_lm_head_model(self, config, input_ids, *args):
        model = CpmBeeForCausalLM(config)
        model.to(torch_device)
        input_ids["input_ids"] = input_ids["input_ids"].to(torch_device)
        model.eval()

        model_output = model(**input_ids)
        self.parent.assertEqual(
            model_output.logits.shape,
            (self.batch_size, self.seq_length, config.vocab_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class CpmBeeModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (CpmBeeModel, CpmBeeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": CpmBeeModel, "text-generation": CpmBeeForCausalLM} if is_torch_available() else {}
    )

    test_pruning = False
    test_missing_keys = False
    test_mismatched_shapes = False
    test_head_masking = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CpmBeeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CpmBeeConfig)

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_inputs_embeds(self):
        unittest.skip("CPMBee doesn't support input_embeds.")(self.test_inputs_embeds)

    def test_retain_grad_hidden_states_attentions(self):
        unittest.skip(
            "CPMBee doesn't support retain grad in hidden_states or attentions, because prompt management will peel off the output.hidden_states from graph.\
                 So is attentions. We strongly recommand you use loss to tune model."
        )(self.test_retain_grad_hidden_states_attentions)

    def test_cpmbee_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cpmbee_model(config, inputs)

    def test_cpmbee_lm_head_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(config, inputs)


@require_torch
class CpmBeeForCausalLMlIntegrationTest(unittest.TestCase):
    @tooslow
    def test_simple_generation(self):
        texts = {"input": "今天天气不错，", "<ans>": ""}
        model = CpmBeeForCausalLM.from_pretrained("openbmb/cpm-bee-10b")
        tokenizer = CpmBeeTokenizer.from_pretrained("openbmb/cpm-bee-10b")
        output_texts = model.generate(texts, tokenizer)
        expected_output = {"input": "今天天气不错，", "<ans>": "适合睡觉。"}
        self.assertEqual(expected_output["<ans>"], output_texts["<ans>"])
