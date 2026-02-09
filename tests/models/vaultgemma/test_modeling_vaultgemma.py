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
"""Testing suite for the PyTorch VaultGemma model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    is_torch_available,
    pipeline,
)
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        VaultGemmaModel,
    )


class VaultGemmaModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = VaultGemmaModel


@require_torch
class VaultGemmaModelTest(CausalLMModelTest, unittest.TestCase):
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = VaultGemmaModelTester


@slow
@require_torch_accelerator
class VaultGemmaIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_bf16(self):
        model_id = "google/vaultgemma-1b"
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on a 1990 240sx. I have a 1",
            "<pad><pad><bos>Hi today I am going to show you how to make a simple 3D model of a 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="eager").to(
            torch_device
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(self.input_text, return_tensors="pt", padding=True).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=False)

        self.assertEqual(output_text, EXPECTED_TEXTS)

    def test_model_pipeline_bf16(self):
        model_id = "google/vaultgemma-1b"
        # EXPECTED_TEXTS should match the same non-pipeline test, minus the special tokens
        EXPECTED_TEXTS = [
            "Hello I am doing a project on a 1990 240sx. I have a 1",
            "Hi today I am going to show you how to make a simple 3D model of a 3D",
        ]

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        output = pipe(self.input_text, max_new_tokens=20, do_sample=False, padding=True)

        self.assertEqual(output[0][0]["generated_text"], EXPECTED_TEXTS[0])
        self.assertEqual(output[1][0]["generated_text"], EXPECTED_TEXTS[1])

    @pytest.mark.torch_export_test
    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.5.0"):
            self.skipTest(reason="This test requires torch >= 2.5 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        model_id = "google/vaultgemma-1b"
        tokenizer = AutoTokenizer.from_pretrained(model_id, pad_token="</s>", padding_side="right")
        EXPECTED_TEXT_COMPLETIONS = Expectations(
            {
                ("cuda", 8): ["Hello I am doing a project on a 1990 240sx. I have a 1"],
            }
        )
        EXPECTED_TEXT_COMPLETION = EXPECTED_TEXT_COMPLETIONS.get_expectation()
        max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
            "input_ids"
        ].shape[-1]

        # Load model
        device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
        dtype = torch.bfloat16
        cache_implementation = "static"
        attn_implementation = "sdpa"
        batch_size = 1
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_config=GenerationConfig(
                use_cache=True,
                cache_implementation=cache_implementation,
                max_length=max_generation_length,
                cache_config={
                    "batch_size": batch_size,
                    "max_cache_len": max_generation_length,
                },
            ),
        )

        prompts = ["Hello I am doing"]
        prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        exportable_module = TorchExportableModuleForDecoderOnlyLM(model)
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=model.device),
            cache_position=torch.tensor([0], dtype=torch.long, device=model.device),
        )
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a hybrid cache).
        Outputs for every attention functions should be coherent and identical.
        """
        # Impossible to test it with this model (even with < 100 tokens), probably due to the compilation of a large model.
        if attn_implementation == "flex_attention":
            self.skipTest(
                reason="`flex_attention` gives `torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 147456 Hardware limit:101376 Reducing block sizes or `num_stages` may help.`"
            )

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

        if torch_device == "xpu" and attn_implementation == "flash_attention_2":
            self.skipTest(reason="Intel XPU doesn't support flash_attention_2 as of now.")

        model_id = "google/vaultgemma-1b"
        EXPECTED_COMPLETIONS = [
            " place pretty place pretty place. place pretty place pretty place. place pretty place pretty place. place pretty",
            ", green, yellow, orange, purple, black, white, and gray.\n\nA list of",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        # It should by Hybrid by default from hub config, but let's make sure!
        out = model.generate(**inputs, max_new_tokens=20, cache_implementation="hybrid")[:, input_size:]
        output_text = tokenizer.batch_decode(out)

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    def test_generation_beyond_sliding_window_dynamic(self, attn_implementation: str):
        """
        Same as above, but explicitly setting the cache to Dynamic, as it's otherwise static by default for
        the model on the hub
        """
        # Impossible to test it with this model (even with < 100 tokens), probably due to the compilation of a large model.
        if attn_implementation == "flex_attention":
            self.skipTest(
                reason="`flex_attention` gives `torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 147456 Hardware limit:101376 Reducing block sizes or `num_stages` may help.`"
            )

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

        if torch_device == "xpu" and attn_implementation == "flash_attention_2":
            self.skipTest(reason="Intel XPU doesn't support flash_attention_2 as of now.")

        model_id = "google/vaultgemma-1b"
        EXPECTED_COMPLETIONS = [
            " place pretty place pretty place. place pretty place pretty place. place pretty place pretty place. place pretty",
            ", green, yellow, orange, purple, black, white, and gray.\n\nA list of",
        ]

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = AutoModelForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=torch.float16
        ).to(torch_device)

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.sliding_window)

        out = model.generate(**inputs, max_new_tokens=20, cache_implementation="dynamic", return_dict_in_generate=True)
        output_text = tokenizer.batch_decode(out.sequences[:, input_size:])

        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

        # Let's check that the dynamic cache has hybrid layers!
        dynamic_cache = out.past_key_values
        self.assertTrue(isinstance(dynamic_cache, DynamicCache))
        for layer, layer_type in zip(dynamic_cache.layers, model.config.layer_types):
            if layer_type == "sliding_attention":
                self.assertTrue(isinstance(layer, DynamicSlidingWindowLayer))
                self.assertEqual(layer.keys.shape[-2], model.config.sliding_window - 1)
            else:
                self.assertTrue(isinstance(layer, DynamicLayer))
                # max_new_tokens - 1 because last token generated is not cached
                self.assertEqual(layer.keys.shape[-2], input_size + 20 - 1)
