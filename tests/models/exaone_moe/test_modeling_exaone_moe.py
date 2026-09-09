# Copyright 2026 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE MoE model."""

import unittest

from pytest import mark

from transformers import (
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        DynamicCache,
        ExaoneMoeConfig,
        ExaoneMoeForCausalLM,
        ExaoneMoeModel,
        StaticCache,
    )
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.exaone_moe.modeling_exaone_moe import ExaoneMoeExperts, ExaoneMoeMLP


class ExaoneMoeModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = ExaoneMoeModel


@require_torch
class ExaoneMoeModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = ExaoneMoeModelTester
    model_split_percents = [0.5, 0.8, 0.9]

    @unittest.skip("ExaoneMoe TP + quantized generation test needs fixing")
    def test_tp_generation_quantized(self):
        pass

    def test_per_layer_sliding_window_cache(self):
        config = ExaoneMoeConfig(
            num_hidden_layers=4,
            layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
            sliding_windows=[0, 4096, 128, 0],
        )

        dynamic_cache = DynamicCache(config=config)
        self.assertEqual(dynamic_cache.layers[1].sliding_window, 4096)
        self.assertEqual(dynamic_cache.layers[2].sliding_window, 128)

        static_cache = StaticCache(config=config, max_cache_len=8192)
        self.assertEqual(static_cache.layers[1].get_max_length(), 4096)
        self.assertEqual(static_cache.layers[2].get_max_length(), 128)

    def test_per_layer_sliding_window_masks(self):
        config = ExaoneMoeConfig(num_hidden_layers=2)
        config._attn_implementation = "eager"
        inputs_embeds = torch.zeros(1, 6, config.hidden_size)
        mask_kwargs = {
            "config": config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": None,
            "allow_is_causal_skip": False,
        }

        short_mask = create_sliding_window_causal_mask(**mask_kwargs, sliding_window=2)
        long_mask = create_sliding_window_causal_mask(**mask_kwargs, sliding_window=4)

        torch.testing.assert_close(short_mask[0, 0, -1] == 0, torch.tensor([False, False, False, False, True, True]))
        torch.testing.assert_close(long_mask[0, 0, -1] == 0, torch.tensor([False, False, True, True, True, True]))

    def test_swiglu_limit_is_applied_before_activation(self):
        config = ExaoneMoeConfig(hidden_size=1, intermediate_size=1, num_experts=1, moe_intermediate_size=1)
        mlp = ExaoneMoeMLP(config, swiglu_limit=1.0)
        experts = ExaoneMoeExperts(config, swiglu_limit=1.0)
        with torch.no_grad():
            mlp.gate_proj.weight.fill_(1.0)
            mlp.up_proj.weight.fill_(1.0)
            mlp.down_proj.weight.fill_(1.0)
            experts.gate_up_proj.fill_(1.0)
            experts.down_proj.fill_(1.0)

        hidden_states = torch.tensor([[10.0]])
        expected = torch.nn.functional.silu(torch.tensor(1.0)).reshape(1, 1)
        torch.testing.assert_close(mlp(hidden_states), expected)
        torch.testing.assert_close(experts(hidden_states, torch.tensor([[0]]), torch.tensor([[1.0]])), expected)


@slow
@require_torch
class ExaoneMoeIntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "hf-internal-testing/EXAONE-MoE-Dummy-7B-A1B"

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = ExaoneMoeForCausalLM.from_pretrained(
                cls.TEST_MODEL_ID,
                device_map="auto",
                experts_implementation="eager",
            )

        return cls.model

    def test_model_logits(self):
        input_ids = [405, 7584, 36608, 892, 95714, 2907, 1492, 758, 373, 582]
        model = self.get_model()
        input_ids = torch.tensor([input_ids]).to(model.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        # fmt: off
        EXPECTED_MEAN = Expectations(
            {
                ("xpu", 5): torch.tensor(
                    [[-2.2315, -3.0607, -3.0840, -3.2559, -3.1974, -3.3996, -3.1582, -3.2548, -3.8877, -0.6911]]
                ),
                ("cuda", None): torch.tensor(
                    [[-2.2491, -3.0824, -3.2191, -3.2712, -3.1991, -3.4087, -3.1384, -3.2601, -3.8869, -0.6940]]
                ),
            }
        ).get_expectation()
        EXPECTED_SLICE = Expectations(
            {
                ("xpu", 5): torch.tensor(
                    [-2.3750, -3.0156, 2.6875, -3.0000, 0.5078, -1.4141, -1.8516, -2.6719, -1.7578, -2.0781]
                ),
                ("cuda", None): torch.tensor(
                    [-2.3906, -3.0469, 2.6875, -3.0156, 0.4941, -1.4219, -1.8672, -2.6719, -1.7656, -2.0938]
                ),
            }
        ).get_expectation()
        # fmt: on
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :10], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)

    def test_model_generation_sdpa(self):
        EXPECTED_TEXT = "The deep learning is 100% accurate.\n\nThe 100% accurate is 100%"
        prompt = "The deep learning is "
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = self.get_model()

        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**input_ids, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        self.assertEqual(EXPECTED_TEXT, text)

    @require_flash_attn
    @mark.flash_attn_test
    def test_model_generation_beyond_sliding_window_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [373, 686, 373, 115708, 373, 885]
        input_ids = [72861, 2711] + [21605, 2711] * 2048
        model = self.get_model()
        model.set_attn_implementation("flash_attention_2")
        input_ids = torch.tensor([input_ids]).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_new_tokens=6, do_sample=False)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-6:].tolist())
