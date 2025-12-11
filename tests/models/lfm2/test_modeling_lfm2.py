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
"""Testing suite for the PyTorch LLaMA model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import Lfm2ForCausalLM, Lfm2Model
    from transformers.models.lfm2.modeling_lfm2 import Lfm2HybridConvCache


class Lfm2ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Lfm2Model

    def __init__(
        self,
        parent,
        layer_types=["full_attention", "conv"],
    ):
        super().__init__(parent)
        self.layer_types = layer_types


@require_torch
class Lfm2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Lfm2ModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Lfm2ForCausalLM if is_torch_available() else None

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        self.assertIsInstance(past_key_values, Lfm2HybridConvCache)

        # (batch, kv heads, seq_length, head_dim)
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        attention_shape = (batch_size, num_heads, seq_length, head_dim)
        conv_shape = (batch_size, config.hidden_size, config.conv_L_cache)

        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == "full_attention":
                self.assertEqual(past_key_values.key_cache[i].shape, attention_shape)
                self.assertEqual(past_key_values.value_cache[i].shape, attention_shape)
            else:
                self.assertEqual(past_key_values.conv_cache[i].shape, conv_shape)

    def _check_caches_are_equal(self, cache1: Lfm2HybridConvCache, cache2: Lfm2HybridConvCache):
        if not isinstance(cache1, Lfm2HybridConvCache) or not isinstance(cache2, Lfm2HybridConvCache):
            raise ValueError("The wrong cache is being used!")

        if not len(cache1) == len(cache2):
            raise ValueError("Both caches do not have the same number of layers.")

        num_layers = len(cache1)
        for idx in range(num_layers):
            torch.testing.assert_close(cache1.key_cache[idx], cache2.key_cache[idx])
            torch.testing.assert_close(cache1.value_cache[idx], cache2.value_cache[idx])
            torch.testing.assert_close(cache1.conv_cache[idx], cache2.conv_cache[idx])

    def test_attention_outputs(self):
        """Lfm2Moe alternates between attention and short-conv layers."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        # force eager attention to support output attentions
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            config = model.config
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
                self_attentions = outputs.attentions

            self.assertEqual(out_len + 1, len(outputs))
            self.assertEqual(len(self_attentions), sum(layer == "full_attention" for layer in config.layer_types))
            self.assertListEqual(list(self_attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])


@require_torch_accelerator
@require_read_token
@slow
class Lfm2IntegrationTest(unittest.TestCase):
    pass
