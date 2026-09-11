# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch FlexOlmo model."""

import unittest

from transformers import is_torch_available
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    Expectations,
    backend_device_count,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        FlexOlmoForCausalLM,
        FlexOlmoModel,
    )


class FlexOlmoModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = FlexOlmoModel

    def __init__(self, parent):
        super().__init__(parent=parent)
        # NOTE(3outeille): must be 0.0 for TP backward tests. In train mode, non-zero dropout causes
        # different RNG states between the non-TP and TP model forward passes (they run sequentially),
        # leading to different dropout masks and mismatched losses.
        self.attention_probs_dropout_prob = 0.0


@require_torch
class FlexOlmoModelTest(CausalLMModelTest, unittest.TestCase):
    test_all_params_have_gradient = False
    model_tester_class = FlexOlmoModelTester

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = FlexOlmoForCausalLM if is_torch_available() else None


@require_torch
class FlexOlmoIntegrationTest(unittest.TestCase):
    model_id = "shanearora/Flex-reddit-2x7B-1T"

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            # Originally (when loading in fp32) device_map="auto" filled all GPUs to ~100%, leaving no
            # room for the ~344 MiB MergeModulelist temporary buffer that fuses per-expert weight shards
            # into a single gate_up_proj tensor during from_pretrained — causing CUDA OOM on multi-GPU.
            # A 70% per-GPU max_memory cap was the fix.
            #
            # We later switched to bfloat16 to fix a separate OOM that occurred during model.generate()
            # after the logits forward pass. With bfloat16 the model footprint is halved (~28 GiB vs
            # ~56 GiB for fp32), so there is naturally enough headroom and the cap is no longer strictly
            # necessary. We keep it here as a marker: the MergeModulelist OOM is a real problem for large
            # MoE models loaded with device_map="auto", and a better automatic solution (e.g. reserving
            # headroom inside the loader itself) would be welcome.
            n = backend_device_count(torch_device)
            if n > 0 and torch_device != "cpu":
                torch_accel = getattr(torch, torch_device)
                per_device = int(
                    min(torch_accel.get_device_properties(i).total_memory for i in range(n)) * 0.70 / 1024**3
                )
                max_memory = dict.fromkeys(range(n), f"{per_device}GiB")
                max_memory["cpu"] = "60GiB"
            else:
                max_memory = None
            cls.model = FlexOlmoForCausalLM.from_pretrained(
                cls.model_id, device_map="auto", max_memory=max_memory, torch_dtype=torch.bfloat16
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "model"):
            del cls.model
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = self.get_model()
        with torch.no_grad():
            out = model(torch.tensor(input_ids, device=model.device)).logits.float()
        # Expected mean on dim = -1
        expectations = Expectations(
            {
                ("cuda", 8): [[-5.4104, -5.3699, -2.3821, -2.1202, -5.9779, -5.4052, -5.4425, -5.8169]],
            }
        )
        EXPECTED_MEAN = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        expectations = Expectations(
            {
                ("cuda", 8): [ 0.5234, -3.6094, -7.2500, -5.0000, -5.8750, -5.2813, -4.2813, -4.6563, -3.4219, -4.6563, -6.5625, -3.1406, -6.0625, -2.1094, -6.4688, -0.5078,  1.2422,  0.7344, -0.1953, -0.4160, -0.6992, -0.9609, -0.9688, -1.3359, -1.2656, -4.5625, -2.4375, -5.5938,  0.7734, -4.5625],
            }
        )  # fmt: skip
        EXPECTED_SLICE = torch.tensor(expectations.get_expectation(), device=torch_device)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the laws of physics are the same in all inertial frames of reference, and 2) the speed of light is constant in all inertial frames of reference. The first statement is called the principle of relativity, and the second is called the constancy of the speed of light. The first statement is"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer", device_map="auto")
        model = self.get_model()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
