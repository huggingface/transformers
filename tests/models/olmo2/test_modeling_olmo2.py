# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch OLMo2 model."""

import unittest

from packaging import version

from transformers import Olmo2Config, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.testing_utils import (
    require_tokenizers,
    require_torch,
    slow,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Olmo2ForCausalLM,
        Olmo2Model,
    )


class Olmo2ModelTester(CausalLMModelTester):
    config_class = Olmo2Config
    if is_torch_available():
        base_model_class = Olmo2Model
        causal_lm_class = Olmo2ForCausalLM


@require_torch
class Olmo2ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (Olmo2Model, Olmo2ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Olmo2Model,
            "text-generation": Olmo2ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    model_tester_class = Olmo2ModelTester

    @unittest.skip(reason="OLMo2 does not support head pruning.")
    def test_headmasking(self):
        pass


@require_torch
class Olmo2IntegrationTest(unittest.TestCase):
    @slow
    def test_model_1b_logits_bfloat16(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = Olmo2ForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B").to(torch.bfloat16)
        out = model(torch.tensor(input_ids)).logits.float()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-5.7094, -6.5548, -3.2527, -2.7847, -5.5092, -4.5223, -4.8427, -4.6867]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([2.4531, -5.7188, -5.1562, -4.8750, -6.7812, -4.0625, -4.4375, -4.5938, -7.5938, -5.0938, -3.9375, -3.6875, -5.0938, -3.1875, -5.6875, 0.2266, 1.2578, 1.1016, 0.8945, 0.4785, 0.2256, -0.3613, -0.4258, 0.1377, -0.1104, -7.1875, -5.2188, -6.8125, -0.9062, -2.9062])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_logits(self):
        input_ids = [[1, 306, 4658, 278, 6593, 310, 2834, 338]]
        model = Olmo2ForCausalLM.from_pretrained("shanearora/OLMo2-7B-1124-hf", device_map="auto")
        out = model(torch.tensor(input_ids)).logits.float()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor(
            [[-13.0244, -13.9564, -11.8270, -11.3047, -12.3794, -12.4215, -15.6030, -12.7962]]
        )
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = torch.tensor([-5.3909, -13.9841, -13.6123, -14.5780, -13.9455, -13.2265, -13.4734, -11.9079, -9.2879, -12.6139, -11.4819, -5.9607, -11.9657, -6.3618, -11.1065, -7.3075, -6.5674, -6.7154, -7.3409, -7.9662, -8.0863, -8.1682, -8.7341, -8.7665, -8.8742, -9.7813, -8.0620, -12.5937, -7.6440, -11.3966])  # fmt: skip
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

    @slow
    def test_model_7b_greedy_generation(self):
        EXPECTED_TEXT_COMPLETION = """Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light is the fastest speed possible, and 3) the speed of light is the same for all observers, regardless of their relative motion. The theory of relativity is based on the idea that the speed of light is constant. This means that"""
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = AutoTokenizer.from_pretrained("shanearora/OLMo2-7B-1124-hf", device_map="auto")
        model = Olmo2ForCausalLM.from_pretrained("shanearora/OLMo2-7B-1124-hf", device_map="auto")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=64, top_p=None, temperature=1, do_sample=False)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @require_tokenizers
    def test_simple_encode_decode(self):
        rust_tokenizer = AutoTokenizer.from_pretrained("shanearora/OLMo2-7B-1124-hf")

        self.assertEqual(rust_tokenizer.encode("This is a test"), [2028, 374, 264, 1296])
        self.assertEqual(rust_tokenizer.decode([2028, 374, 264, 1296], skip_special_tokens=True), "This is a test")

        # bytefallback showcase
        self.assertEqual(rust_tokenizer.encode("生活的真谛是"), [21990, 76706, 9554, 89151, 39013, 249, 21043])  # fmt: skip
        self.assertEqual(
            rust_tokenizer.decode([21990, 76706, 9554, 89151, 39013, 249, 21043], skip_special_tokens=True),
            "生活的真谛是",
        )

        # Inner spaces showcase
        self.assertEqual(rust_tokenizer.encode("Hi  Hello"), [13347, 220, 22691])
        self.assertEqual(rust_tokenizer.decode([13347, 220, 22691], skip_special_tokens=True), "Hi  Hello")

        self.assertEqual(rust_tokenizer.encode("Hi   Hello"), [13347, 256, 22691])
        self.assertEqual(rust_tokenizer.decode([13347, 256, 22691], skip_special_tokens=True), "Hi   Hello")

        self.assertEqual(rust_tokenizer.encode(""), [])

        self.assertEqual(rust_tokenizer.encode(" "), [220])

        self.assertEqual(rust_tokenizer.encode("  "), [256])

        self.assertEqual(rust_tokenizer.encode(" Hello"), [22691])

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        olmo2_model = "shanearora/OLMo2-7B-1124-hf"

        tokenizer = AutoTokenizer.from_pretrained(olmo2_model, pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is constant, 2) the speed of light",
        ]
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation=cache_implementation,
            max_length=max_generation_length,
            cache_config={
                "batch_size": batch_size,
                "max_cache_len": max_generation_length,
            },
        )
        model = Olmo2ForCausalLM.from_pretrained(
            olmo2_model,
            device_map=device,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=generation_config,
        )

        prompts = ["Simply put, the theory of relativity states that "]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + eager
        eager_generated_ids = model.generate(
            **prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, cache_implementation=cache_implementation
        )
        eager_generated_text = tokenizer.batch_decode(eager_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, eager_generated_text)

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
