# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HyperCLOVAX model."""

import unittest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        HyperCLOVAXConfig,
        HyperCLOVAXForCausalLM,
        HyperCLOVAXModel,
    )


class HyperCLOVAXModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = HyperCLOVAXModel


@require_torch
class HyperCLOVAXModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HyperCLOVAXModelTester

    # Same as Granite — avoids edge cases with the causal_mask buffer during CPU offload
    model_split_percents = [0.5, 0.7, 0.8]

    _torch_compile_train_cls = HyperCLOVAXForCausalLM if is_torch_available() else None

    def test_mup_attention_scaling(self):
        """Changing attention_multiplier must produce different logits."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Baseline: default attention_multiplier (head_dim ** -0.5)
        model_default = HyperCLOVAXForCausalLM(config).to(torch_device).eval()

        # Modified: use a clearly different multiplier
        config_scaled = HyperCLOVAXConfig(**config.to_dict())
        config_scaled.attention_multiplier = config.attention_multiplier * 2.0
        model_scaled = HyperCLOVAXForCausalLM(config_scaled).to(torch_device).eval()

        # Copy weights so the only difference is the scaling factor
        model_scaled.load_state_dict(model_default.state_dict())

        input_ids = inputs_dict["input_ids"].to(torch_device)
        with torch.no_grad():
            logits_default = model_default(input_ids).logits
            logits_scaled = model_scaled(input_ids).logits

        self.assertFalse(
            torch.allclose(logits_default, logits_scaled),
            "attention_multiplier has no effect on logits — MuP attention scaling is broken.",
        )

    def test_mup_logits_scaling(self):
        """Changing logits_scaling must produce proportionally scaled logits."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        scale_factor = 2.0
        config_scaled = HyperCLOVAXConfig(**config.to_dict())
        config_scaled.logits_scaling = config.logits_scaling * scale_factor
        # Keep attention_multiplier identical so only the final logit scale changes
        config_scaled.attention_multiplier = config.attention_multiplier

        model_default = HyperCLOVAXForCausalLM(config).to(torch_device).eval()
        model_scaled = HyperCLOVAXForCausalLM(config_scaled).to(torch_device).eval()
        model_scaled.load_state_dict(model_default.state_dict())

        input_ids = inputs_dict["input_ids"].to(torch_device)
        with torch.no_grad():
            logits_default = model_default(input_ids).logits
            logits_scaled = model_scaled(input_ids).logits

        self.assertTrue(
            torch.allclose(logits_scaled, logits_default * scale_factor, atol=1e-4),
            "logits_scaling does not scale logits proportionally — MuP logit scaling is broken.",
        )

    def test_post_norm_output_shape(self):
        """use_post_norm=True must not change the output shape."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model_no_post_norm = HyperCLOVAXForCausalLM(config).to(torch_device).eval()

        config_post_norm = HyperCLOVAXConfig(**config.to_dict())
        config_post_norm.use_post_norm = True
        model_post_norm = HyperCLOVAXForCausalLM(config_post_norm).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"].to(torch_device)
        with torch.no_grad():
            out_default = model_no_post_norm(input_ids).logits
            out_post_norm = model_post_norm(input_ids).logits

        self.assertEqual(
            out_default.shape,
            out_post_norm.shape,
            "use_post_norm=True changes the output shape unexpectedly.",
        )

    def test_post_norm_changes_output(self):
        """use_post_norm=True must produce different outputs than the default (no post-norm)."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model_no_post_norm = HyperCLOVAXForCausalLM(config).to(torch_device).eval()

        config_post_norm = HyperCLOVAXConfig(**config.to_dict())
        config_post_norm.use_post_norm = True
        model_post_norm = HyperCLOVAXForCausalLM(config_post_norm).to(torch_device).eval()

        input_ids = inputs_dict["input_ids"].to(torch_device)
        with torch.no_grad():
            out_default = model_no_post_norm(input_ids).logits
            out_post_norm = model_post_norm(input_ids).logits

        self.assertFalse(
            torch.allclose(out_default, out_post_norm),
            "use_post_norm=True produces identical outputs to the default — Peri-Layer Norm has no effect.",
        )


@slow
@require_torch_accelerator
class HyperCLOVAXIntegrationTest(unittest.TestCase):
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
    input_text = ["서울에서 부산까지 기차로 걸리는 시간은 ", "The travel time by train from Seoul to Busan"]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_seed_think_14b_logits_bf16(self):
        # tokenizer.encode("대한민국의 수도는 서울입니다.", add_special_tokens=True)
        LOGIT_INPUT_IDS = [105319, 21028, 107115, 16969, 102949, 80052, 13]

        # fmt: off
        expected_means = Expectations(
            {
                ("cuda", None): torch.tensor([[-1.0737, -5.0637, 0.3728, -2.9377, 2.1582, 2.8907, -3.0403]]),
            }
        )
        expected_slices = Expectations(
            {
                ("cuda", None): torch.tensor([[3.0156, 3.8438, 3.0625, 3.7344, 3.1250, 2.6406, 4.5625, 5.6563, 5.0000, 4.0000, 4.3750, 6.3125, 5.6250, 5.4375, 5.4375]]),
            }
        )
        # fmt: on

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
        )
        with torch.no_grad():
            out = model(torch.tensor([LOGIT_INPUT_IDS]).to(torch_device))

        expected_mean = expected_means.get_expectation().to(torch_device)
        self.assertTrue(torch.allclose(out.logits.float().mean(-1), expected_mean, atol=1e-2, rtol=1e-2))

        expected_slice = expected_slices.get_expectation().to(torch_device)
        self.assertTrue(torch.allclose(out.logits[0, 0, :15].float(), expected_slice, atol=1e-2, rtol=1e-2))

    @require_torch_large_accelerator
    def test_model_seed_think_14b_bf16(self):
        # input_text[0]: Korean, input_text[1]: English — covers both languages
        EXPECTED_TEXTS = [
            "서울에서 부산까지 기차로 걸리는 시간은 2시간 30분에서 3시간 사이입니다. 기차 종류에 따라 시간이 달라질",
            "The travel time by train from Seoul to Busan is approximately 2.5 to 3 hours, depending on the type of train. The K",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="eager"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    @require_torch_large_accelerator
    def test_model_seed_think_14b_bf16_sdpa(self):
        # input_text[0]: Korean, input_text[1]: English — covers both languages
        EXPECTED_TEXTS = [
            "서울에서 부산까지 기차로 걸리는 시간은 2시간 30분에서 3시간 사이입니다. 기차 종류에 따라 시간이 달라질",
            "The travel time by train from Seoul to Busan is approximately 2.5 to 3 hours, depending on the type of train. The K",
        ]

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)
