# coding=utf-8
# Copyright 2022 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CPMAnt model. """

import unittest

from transformers.testing_utils import is_torch_available, require_torch, slow

from ...generation.test_utils import torch_device
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST,
        CPMAntConfig,
        CPMAntForCausalLM,
        CPMAntModel,
        CPMAntTokenizer,
    )


class CPMAntConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "dim_model"))
        self.parent.assertTrue(hasattr(config, "num_heads"))
        self.parent.assertTrue(hasattr(config, "dim_head"))


@require_torch
class CPMAntModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=8,
        is_training=True,
        use_token_type_ids=False,
        use_input_mask=True,
        use_labels=True,
        use_mc_token_ids=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
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
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = dict()
        input_ids["input_ids"] = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).type(torch.int32)

        config = self.get_config()

        return (config, input_ids)

    def get_config(self):
        return CPMAntConfig.from_pretrained("openbmb/cpm-ant-10b")

    def create_and_check_cpmant_model(self, config, input_ids, *args):
        model = CPMAntModel(config=config)
        model.to(torch_device)
        model.eval()

        hidden_states = model(**input_ids).last_hidden_state

        self.parent.assertEqual(hidden_states.shape, (self.batch_size, self.seq_length, config.dim_model))

    def create_and_check_lm_head_model(self, config, input_ids, *args):
        model = CPMAntForCausalLM(config)
        model.to(torch_device)
        for k, v in input_ids.items():
            input_ids[k] = v.to(torch_device)
        model.eval()

        model_output = model(**input_ids)
        self.parent.assertEqual(model_output.logits.shape, (self.batch_size, self.seq_length, config.vocab_size + config.prompt_types * config.prompt_length))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch
class CPMAntModelTest(unittest.TestCase):
    all_model_classes = (
        (
            CPMAntModel,
            CPMAntForCausalLM,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = CPMAntModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CPMAntConfig)

    def test_config(self):
        self.create_and_test_config_common_properties()
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def create_and_test_config_common_properties(self):
        return

    @slow
    def test_cpmant_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_cpmant_model(config, inputs)

    @slow
    def test_cpmant_lm_head_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(config, inputs)

    @slow
    def test_simple_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntForCausalLM.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        texts = "今天天气不错，"
        expected_output = "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的"
        model_inputs = tokenizer(texts, return_tensors="pt")
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.decode(token_ids)
        self.assertEqual(expected_output, output_texts)

    @slow
    def test_batch_generation(self):
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntForCausalLM.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        texts = ["今天天气不错，", "新年快乐，万事如意！"]
        expected_output = [
            "今天天气不错，阳光明媚，我和妈妈一起去超市买东西。\n在超市里，我看到了一个很好玩的玩具，它的名字叫“机器人”。它有一个圆圆的脑袋，两只圆圆的眼睛，还有一个圆圆的",
            "新年快乐，万事如意！在这辞旧迎新的美好时刻，我谨代表《农村新技术》杂志社全体同仁，向一直以来关心、支持《农村新技术》杂志发展的各级领导、各界朋友和广大读者致以最诚挚的",
        ]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        token_ids = model.generate(**model_inputs)
        output_texts = tokenizer.batch_decode(token_ids)
        self.assertEqual(expected_output, output_texts)


@require_torch
class CPMAntModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntModel.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).last_hidden_state

        expected_slice = torch.tensor(
            [[[  6.1708,   5.9244,   1.0835], [  6.5207,   6.2893, -11.3324], [ -1.0107,  -0.0576,  -5.9577]]],
        )
        self.assertTrue(torch.allclose(hidden_states[:, :3, :3], expected_slice, atol=1e-2))


@require_torch
class CPMAntForCausalLMlIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_casual(self):
        texts = "今天天气真好！"
        model_path = "openbmb/cpm-ant-10b"
        model = CPMAntForCausalLM.from_pretrained(model_path)
        tokenizer = CPMAntTokenizer.from_pretrained(model_path)
        inputs = tokenizer(texts, return_tensors="pt")
        hidden_states = model(**inputs).logits

        expected_slice = torch.tensor(
            [[[-6.4267, -6.4083, -6.3958], [-5.8802, -5.9447, -5.7811], [-5.3896, -5.4820, -5.4295]]],
        )
        self.assertTrue(torch.allclose(hidden_states[:, :3, :3], expected_slice, atol=1e-2))
