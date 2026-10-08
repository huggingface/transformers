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

import unittest
from unittest import mock

from transformers import is_torch_available
from transformers.testing_utils import is_flash_linear_attention_available, require_torch, torch_device

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch

    from transformers import BailingHybridForCausalLM, BailingHybridModel, DynamicCache
    from transformers.generation.utils import ALL_CACHE_NAMES


class BailingHybridModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = BailingHybridModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.attention_probs_dropout_prob = 0.0
        self.hidden_act = "silu"
        self.num_hidden_layers = 2
        self.layer_types = ["linear_attention", "full_attention"]
        self.first_k_dense_replace = 1
        self.short_conv_kernel_size = 2
        self.head_dim = 16
        self.q_lora_rank = None
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
        self.gated_attention_proj_granularity_type = "head_wise"
        self.num_nextn_predict_layers = 0


@require_torch
class BailingHybridModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = BailingHybridModelTester
    model_tester: BailingHybridModelTester
    has_attentions = False

    def _get_conv_state_shape(self, batch_size: int, config):
        return (batch_size, 3 * config.num_attention_heads * config.linear_head_dim, config.linear_conv_kernel_dim)

    @unittest.skipIf(
        is_flash_linear_attention_available(),
        "FLA disables compilation inside the fused recurrent KDA decode kernel",
    )
    def test_generate_compile_model_forward_fullgraph(self):
        super().test_generate_compile_model_forward_fullgraph()

    @unittest.skip(
        "Packed recurrent cache tensors are recreated between generate calls, which currently recompiles the "
        "compiled decode graph (the same limitation exists in Kimi Linear)."
    )
    def test_static_cache_no_recompile_with_smaller_length(self):
        pass

    def _get_recurrent_state_shape(self, batch_size: int, config):
        return (batch_size, config.num_attention_heads, config.linear_head_dim, config.linear_head_dim)

    @unittest.skip("The specific cache format cannot be instantiated from dp/ddp data.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("MLA uses different query/key and value head dimensions.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(
        "Ling checkpoint conversion is intentionally two-pass: `.attention.` is first renamed to `.self_attn.`, "
        "then forget-gate parameters are moved below `.forget_gate.`. The generic reverse-mapping assertion "
        "cannot represent this ordering; save/load round-trip tests cover the supported behavior."
    )
    def test_reverse_loading_mapping(self):
        pass

    def test_hybrid_layer_pattern(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = BailingHybridModel(config)
        self.assertEqual(model.config.layer_types, ["linear_attention", "full_attention"])
        self.assertEqual(model.layers[0].self_attn.__class__.__name__, "BailingHybridKimiDeltaAttention")
        self.assertEqual(model.layers[1].self_attn.__class__.__name__, "BailingHybridAttention")

    def test_cached_decode_matches_full_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = BailingHybridForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor((1, 8), config.vocab_size).to(torch_device)

        with torch.no_grad():
            full_logits = model(input_ids=input_ids).logits
            cache = DynamicCache(config=config)
            cached_logits = []
            for token in input_ids.split(1, dim=1):
                cached_logits.append(model(input_ids=token, past_key_values=cache, use_cache=True).logits)

        tol = 1e-3 if is_flash_linear_attention_available() else 1e-4
        torch.testing.assert_close(torch.cat(cached_logits, dim=1), full_logits, rtol=tol, atol=tol)

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
