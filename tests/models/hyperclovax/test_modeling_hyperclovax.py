# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
    is_tensor_parallel_test,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
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

    @unittest.skip(
        "In TP mode, Float8 quantization derives scales per shard rather than globally, "
        "so each TP rank observes different weight magnitudes than the full-weight non-TP "
        "baseline. HyperCLOVAX's Peri-Layer Normalization (post_norm1/post_norm2) amplifies "
        "this discrepancy past the 75% token-match threshold. Skipped pending an upstream fix."
    )
    @is_tensor_parallel_test
    def test_tp_generation_quantized(self):
        pass


@slow
@require_torch
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
                ("cuda", (8, 6)): torch.tensor([[-1.0764, -5.0859,  0.3363, -2.9254,  2.1648,  2.9170, -2.9659]]),
            }
        ).get_expectation().to(torch_device)

        expected_slices = Expectations(
            {
                ("cuda", None): torch.tensor([3.0156, 3.8438, 3.0625, 3.7344, 3.1250, 2.6406, 4.5625, 5.6563, 5.0000, 4.0000, 4.3750, 6.3125, 5.6250, 5.4375, 5.4375]),
                ("cuda", (8, 6)): torch.tensor([3.0156, 3.8594, 3.0781, 3.7500, 3.1406, 2.6406, 4.5625, 5.6562, 5.0000, 4.0000, 4.3750, 6.3125, 5.6250, 5.4375, 5.4375]),
            }
        ).get_expectation().to(torch_device)
        # fmt: on

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        with torch.no_grad():
            out = model(torch.tensor([LOGIT_INPUT_IDS]).to(torch_device))

        self.assertTrue(torch.allclose(out.logits.float().mean(-1), expected_means, atol=1e-2, rtol=1e-2))
        self.assertTrue(torch.allclose(out.logits[0, 0, :15].float(), expected_slices, atol=1e-2, rtol=1e-2))

    def test_model_seed_think_14b_bf16(self):
        # input_text[0]: Korean, input_text[1]: English — covers both languages
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", None): [
                    "서울에서 부산까지 기차로 걸리는 시간은 2시간 30분에서 3시간 사이입니다. 기차 종류에 따라 시간이 달라질",
                    "The travel time by train from Seoul to Busan is approximately 2.5 to 3 hours, depending on the type of train. The K",
                ],
            }
        ).get_expectation()

        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)
