# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Ling 3.0 / Bailing MoE V3 model."""

import tempfile
import unittest
from unittest import mock

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from transformers import BailingHybridForCausalLM, BailingHybridModel
    from transformers.generation.utils import ALL_CACHE_NAMES


class BailingHybridModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BailingHybridModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.hidden_act = "silu"
        self.num_hidden_layers = 2
        self.layer_types = ["linear_attention", "full_attention"]
        self.first_k_dense_replace = 1
        self.short_conv_kernel_size = 2
        # Keep the KDA head size distinct from the MLA RoPE size to cover config save/load semantics.
        self.head_dim = 24
        self.kv_lora_rank = 16
        self.qk_nope_head_dim = 32
        self.qk_rope_head_dim = 16
        self.v_head_dim = 32
        self.moe_intermediate_size = 16
        self.num_local_experts = 4
        self.n_shared_experts = 1
        self.num_experts_per_tok = 2
        self.n_group = 1
        self.topk_group = 1
        self.num_mtp_layers = 0


@require_torch
class BailingHybridModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BailingHybridModelTester
    model_tester: BailingHybridModelTester

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.num_attention_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_attention_heads, config.linear_head_dim, config.linear_head_dim)

    def test_attention_outputs(self):
        """Only full-attention layers produce attention maps in this hybrid architecture."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        config._attn_implementation = "eager"
        seq_len = getattr(self.model_tester, "seq_length", None)
        expected_num_attentions = sum(layer == "full_attention" for layer in config.layer_types)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)

            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)
            self.assertListEqual(list(attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len])
            output_len = len(outputs)

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(output_len + 1, len(outputs))
            self.assertEqual(len(outputs.attentions), expected_num_attentions)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [config.num_attention_heads, seq_len, seq_len]
            )

    def test_reverse_loading_mapping(self):
        """Exercise the two-stage Ling key mapping as an end-to-end exact round trip."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = BailingHybridForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            saved_keys = set(load_file(f"{tmpdirname}/model.safetensors"))
            reloaded = BailingHybridForCausalLM.from_pretrained(tmpdirname)

        expected_keys = {
            "model.word_embeddings.weight",
            "model.layers.0.attention.f_proj.weight",
            "model.layers.0.attention.q_conv1d.weight",
            "model.layers.0.attention.k_conv1d.weight",
            "model.layers.0.attention.v_conv1d.weight",
            "model.layers.1.attention.dense.weight",
            "model.layers.1.mlp.gate.expert_bias",
        }
        self.assertTrue(expected_keys.issubset(saved_keys))
        self.assertNotIn("model.layers.0.self_attn.conv1d.weight", saved_keys)
        state_dict = model.state_dict()
        reloaded_state_dict = reloaded.state_dict()
        self.assertNotEqual(config.linear_head_dim, config.qk_rope_head_dim)
        self.assertEqual(reloaded.config.linear_head_dim, config.linear_head_dim)
        self.assertEqual(reloaded.config.qk_rope_head_dim, config.qk_rope_head_dim)
        self.assertEqual(state_dict.keys(), reloaded_state_dict.keys())
        for key, value in state_dict.items():
            torch.testing.assert_close(value, reloaded_state_dict[key], rtol=0, atol=0)

    def test_recurrent_layers_mask_padding_on_continued_forward(self):
        # Ling's MoE and KDA chunk boundaries amplify harmless split-vs-full accumulation differences to roughly
        # 1e-5--1e-4. Keep the threshold well below the ~1e-3 padding-state contamination this regression targets.
        with mock.patch("transformers.utils.import_utils.is_torchdynamo_compiling", return_value=True):
            for model_class in self.all_generative_model_classes:
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                model = model_class(config).to(torch_device).eval()
                input_ids = inputs_dict["input_ids"][:2].to(torch_device)

                pad_token_id = 7
                seq_len = input_ids.shape[1]
                turn1_len = seq_len // 2 + 1
                pad_len = max(1, seq_len - turn1_len - 1)
                turn1, turn2 = input_ids[:, :turn1_len], input_ids[:, turn1_len:].clone()
                turn2[1, :pad_len] = pad_token_id
                attention_mask = torch.ones_like(input_ids)
                attention_mask[1, turn1_len : turn1_len + pad_len] = 0
                position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

                with torch.no_grad():
                    single = model(
                        input_ids=torch.cat([turn1, turn2], dim=-1),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    ).logits
                    out1 = model(
                        input_ids=turn1,
                        attention_mask=attention_mask[:, :turn1_len],
                        position_ids=position_ids[:, :turn1_len],
                        use_cache=True,
                    )
                    cache_kwarg, cache = next(
                        (
                            (name, getattr(out1, name))
                            for name in ALL_CACHE_NAMES
                            if getattr(out1, name, None) is not None
                        ),
                        ("past_key_values", None),
                    )
                    out2 = model(
                        input_ids=turn2,
                        attention_mask=attention_mask,
                        position_ids=position_ids[:, turn1_len:],
                        use_cache=True,
                        **{cache_kwarg: cache},
                    )

                torch.testing.assert_close(out2.logits[:, -1], single[:, -1], rtol=1e-3, atol=1e-4)

    @unittest.skip("MTP assisted generation cannot roll back the stateful KDA cache.")
    def test_generate_with_mtp(self):
        pass

    @unittest.skip("MLA uses different query/key and value head dimensions, so Flash SDPA cannot dispatch.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass
