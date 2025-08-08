# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Persimmon model."""

import gc
import unittest

from transformers import PersimmonConfig, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    require_bitsandbytes,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        PersimmonForCausalLM,
        PersimmonForSequenceClassification,
        PersimmonForTokenClassification,
        PersimmonModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class PersimmonModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = PersimmonConfig
        base_model_class = PersimmonModel
        causal_lm_class = PersimmonForCausalLM
        sequence_class = PersimmonForSequenceClassification
        token_class = PersimmonForTokenClassification


@require_torch
class PersimmonModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = PersimmonModelTester
    all_model_classes = (
        (PersimmonModel, PersimmonForCausalLM, PersimmonForSequenceClassification, PersimmonForTokenClassification)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": PersimmonModel,
            "text-classification": PersimmonForSequenceClassification,
            "token-classification": PersimmonForTokenClassification,
            # TODO (ydshieh): check why these two fail. Fix them or skip them in a better way.
            # "text-generation": PersimmonForCausalLM,
            # "zero-shot": PersimmonForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = PersimmonModelTester

    test_headmasking = False
    test_pruning = False

    @unittest.skip("Persimmon applies key/query norm which doesn't work with packing")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Persimmon applies key/query norm which doesn't work with packing")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


@require_torch
class PersimmonIntegrationTest(unittest.TestCase):
    @slow
    @require_torch_accelerator
    @require_bitsandbytes
    def test_model_8b_chat_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = PersimmonForCausalLM.from_pretrained(
            "adept/persimmon-8b-chat", load_in_8bit=True, device_map={"": 0}, dtype=torch.float16
        )
        out = model(torch.tensor([input_ids], device=torch_device)).logits.float()

        EXPECTED_MEAN = torch.tensor(
            [[-11.4726, -11.1495, -11.2694, -11.2223, -10.9452, -11.0663, -11.0031, -11.1028]]
        )
        # change dtype to `torch.float32` before calling `mean` to avoid `nan` values
        torch.testing.assert_close(out.cpu().to(torch.float32).mean(-1), EXPECTED_MEAN, rtol=1e-4, atol=1e-4)
        # fmt: off
        EXPECTED_SLICE = torch.tensor(
            [-16.9062, -16.9062, -16.9062, -16.9062, -16.8906, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9062, -16.9531, -16.9062, -16.9531, -16.9062, -16.9062],
            dtype=torch.float16
        )
        # fmt: on
        torch.testing.assert_close(out.cpu()[0, 0, :30], EXPECTED_SLICE, rtol=1e-5, atol=1e-5)

        backend_empty_cache(torch_device)
        del model
        gc.collect()

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    @require_bitsandbytes
    def test_model_8b_chat_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """human: Simply put, the theory of relativity states that?\n\nadept: The theory of relativity states that the laws of physics are the same for all observers, regardless of their relative motion."""
        prompt = "human: Simply put, the theory of relativity states that?\n\nadept:"
        tokenizer = AutoTokenizer.from_pretrained("adept/persimmon-8b-chat", use_fast=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch_device)
        model = PersimmonForCausalLM.from_pretrained(
            "adept/persimmon-8b-chat", load_in_8bit=True, device_map={"": 0}, dtype=torch.float16
        )

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        backend_empty_cache(torch_device)
        del model
        gc.collect()
