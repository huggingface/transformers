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

import pytest
from packaging import version

from transformers import AutoTokenizer, StaticCache, is_torch_available
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    run_test_using_subprocess,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        LlamaForQuestionAnswering,
        LlamaForSequenceClassification,
        LlamaForTokenClassification,
        LlamaModel,
        LlamaTokenizer,
    )
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class LlamaModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = LlamaConfig
        base_model_class = LlamaModel
        causal_lm_class = LlamaForCausalLM
        sequence_class = LlamaForSequenceClassification
        token_class = LlamaForTokenClassification


@require_torch
class LlamaModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            LlamaModel,
            LlamaForCausalLM,
            LlamaForSequenceClassification,
            LlamaForQuestionAnswering,
            LlamaForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": LlamaModel,
            "text-classification": LlamaForSequenceClassification,
            "text-generation": LlamaForCausalLM,
            "zero-shot": LlamaForSequenceClassification,
            "question-answering": LlamaForQuestionAnswering,
            "token-classification": LlamaForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = LlamaModelTester
    rotary_embedding_layer = LlamaRotaryEmbedding  # Enables RoPE tests if set

    # Need to use `0.8` instead of `0.9` for `test_cpu_offload`
    # This is because we are hitting edge cases with the causal_mask buffer
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = LlamaForCausalLM if is_torch_available() else None


@require_torch_accelerator
@require_read_token
class LlamaIntegrationTest(unittest.TestCase):
    def setup(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain tests a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_llama_3_1_hard(self):
        """
        An integration test for llama 3.1. It tests against a long output to ensure the subtle numerical differences
        from llama 3.1.'s RoPE can be detected
        """
        expected_texts = Expectations(
            {
                ("rocm", (9, 5)): 'Tell me about the french revolution. The french revolution was a period of radical social and political upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative assembly that had not met since 1614. The Third Estate, which represented the common people, demanded greater representation and eventually broke away to form the National Assembly. This marked the beginning of the end of the absolute monarchy and the rise of the middle class.\n',
                ("cuda", None): 'Tell me about the french revolution. The french revolution was a period of radical political and social upheaval in France that lasted from 1789 until 1799. It was a time of great change and upheaval, marked by the overthrow of the monarchy, the rise of the middle class, and the eventual establishment of the First French Republic.\nThe revolution began in 1789 with the Estates-General, a representative assembly that had not met since 1614. The Third Estate, which represented the common people, demanded greater representation and eventually broke away to form the National Assembly. The National Assembly adopted the Declaration of the Rights of Man and of the Citizen, which enshr',
            }
        )  # fmt: skip
        EXPECTED_TEXT = expected_texts.get_expectation()

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto", dtype=torch.bfloat16
        )
        input_text = ["Tell me about the french revolution."]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    @slow
    def test_model_7b_logits_bf16(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))
        # Expected mean on dim = -1

        # fmt: off
        expected_means = Expectations(
            {
            ("xpu", 3): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
            ("cuda", 7): torch.tensor([[-6.5061, -4.1147, -4.9669, -3.2038, 0.8069, -2.9694, 1.2864, -3.3786]]),
            ("cuda", 8): torch.tensor([[-6.5208, -4.1218, -4.9377, -3.2536,  0.8127, -2.9811,  1.2918, -3.3848]]),
            ("rocm", (9, 4)): torch.tensor([[-6.5094, -4.1329, -4.9754, -3.5042,  0.8082, -2.9443,  1.2830, -3.3539]]),
        })

        expected_mean = expected_means.get_expectation().to(torch_device)
        actual_mean = out.logits.float().mean(-1)
        self.assertTrue(
            torch.allclose(
                expected_mean,
                actual_mean,
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
            ("xpu", 3): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]]),
            ("cuda", 7): torch.tensor([[-12.5000, -7.0625, -0.6289, -7.8750, -6.9688, -7.8125, -6.4688, -7.4375, -7.6875, -6.9375, -6.0312, -7.0000, -1.8594, 1.8438, -8.5000]]),
            ("cuda", 8): torch.tensor([[-12.5625,  -7.1250,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9688,  -6.0312,  -7.0312,  -1.8203,   1.8750, -8.5000]]),
            ("rocm", (9, 4)): torch.tensor([[-12.5000,  -7.0625,  -0.6289,  -7.8750,  -6.9688,  -7.8125,  -6.5000, -7.4375,  -7.6562,  -6.9375,  -6.0312,  -7.0312,  -1.8594,   1.8438, -8.5000]])
        })
        # fmt: on
        expected_slice = expected_slices.get_expectation().to(torch_device)
        actual_slice = out.logits[0, 0, :15].float()
        self.assertTrue(torch.allclose(expected_slice, actual_slice, atol=1e-2, rtol=1e-2))

    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto", dtype=torch.float16)

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(torch_device))

        # fmt: off
        # Expected mean on dim = -1
        expected_means = Expectations(
          {
            ("xpu", 3): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
            ("cuda", 7): torch.tensor([[-6.6420, -4.1227, -4.9809, -3.2041, 0.8261, -3.0052, 1.2957, -3.3648]]),
            ("cuda", 8): torch.tensor([[-6.6544, -4.1259, -4.9840, -3.2456,  0.8261, -3.0124,  1.2971, -3.3641]]),
        })

        expected_mean = expected_means.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_mean.to(torch_device),
                out.logits.float().mean(-1),
                atol=1e-2,
                rtol=1e-2
            )
        )

        # slicing logits[0, 0, 0:15]
        expected_slices = Expectations(
            {
              ("xpu", 3): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328]),
              ("cuda", 7): torch.tensor([-12.8125, -7.3359, -0.4846, -8.0234, -7.2383, -7.9922, -6.4805, -7.7344, -7.8125, -7.0078, -6.1797, -7.1094, -1.8633, 1.9736, -8.6016]),
              ("cuda", 8): torch.tensor([-12.8281,  -7.4609,  -0.4668,  -8.0703,  -7.2539,  -8.0078,  -6.4961, -7.7734,  -7.8516,  -7.0352,  -6.2188,  -7.1367,  -1.8564,   1.9922, -8.6328])
        })
        # fmt: on

        expected_slice = expected_slices.get_expectation()
        self.assertTrue(
            torch.allclose(
                expected_slice.to(torch_device),
                out.logits[0, 0, :15].float(),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    # TODO joao, manuel: remove this in v4.62.0
    # TODO: check why we have the following strange situation.
    # without running in subprocess, this test causes subsequent tests failing with `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`
    @run_test_using_subprocess
    @slow
    def test_model_7b_dola_generation(self):
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        EXPECTED_TEXT_COMPLETION = (
            "Simply put, the theory of relativity states that 1) time and space are relative, and 2) the laws of "
            "physics are the same for all observers in uniform motion relative to one another.\n\nThe theory of "
            "relativity was developed by Albert Einstein in the early 20th century, and it revolutionized our "
            "understanding of space and time."
        )
        prompt = "Simply put, the theory of relativity states that "
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", device_map="sequential", dtype=torch.float16
        )
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # greedy generation outputs
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=64,
            top_p=None,
            temperature=1,
            do_sample=False,
            dola_layers="low",
            trust_remote_code=True,
            custom_generate="transformers-community/dola",
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        NUM_TOKENS_TO_GENERATE = 40
        # Note on `EXPECTED_TEXT_COMPLETION`'s diff: the current value matches the original test if the original test
        # was changed to have a cache of 53 tokens (as opposed to 4096), on Ampere GPUs.
        EXPECTED_TEXT_COMPLETION = [
            "Simply put, the theory of relativity states that 1) the speed of light is constant in all inertial "
            "reference frames, and 2) the laws of physics are the same for all inertial reference frames.\nThe "
            "theory of relativ",
            "My favorite all time favorite condiment is ketchup. I love it on everything. I love it on my eggs, "
            "my fries, my chicken, my burgers, my hot dogs, my sandwiches, my salads, my p",
        ]

        prompts = [
            "Simply put, the theory of relativity states that ",
            "My favorite all time favorite condiment is ketchup.",
        ]
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", pad_token="</s>", padding_side="right")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf", device_map=torch_device, dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache + compile (`generate()` internally compiles each decoding step when static cache is used)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

    @slow
    @pytest.mark.torch_export_test
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        llama_models = {
            "meta-llama/Llama-3.2-1B": [
                "Simply put, the theory of relativity states that 1) the speed of light is the same for all "
                "observers, regardless of their location, and 2) the laws of physics are the same for all observers"
            ],
        }

        for llama_model_ckp, EXPECTED_TEXT_COMPLETION in llama_models.items():
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(llama_model_ckp, pad_token="</s>", padding_side="right")
            max_generation_length = tokenizer(EXPECTED_TEXT_COMPLETION, return_tensors="pt", padding=True)[
                "input_ids"
            ].shape[-1]

            # Load model
            device = "cpu"  # TODO (joao / export experts): should be on `torch_device`, but causes GPU OOM
            dtype = torch.bfloat16
            cache_implementation = "static"
            attn_implementation = "sdpa"
            batch_size = 1
            model = LlamaForCausalLM.from_pretrained(
                llama_model_ckp,
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
                        "device": device,
                    },
                ),
            )

            prompts = ["Simply put, the theory of relativity states that "]
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


@slow
@require_torch_accelerator
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model_dtype = torch.float32
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, dtype=self.model_dtype).to(torch_device)

    def get_test_data(self):
        template = "my favorite {}"
        items = ("pet is a", "artist plays a", "name is L")  # same number of tokens in each item

        batch_separate = [template.format(x) for x in items]  # 3 separate lines
        batch_shared_prefix = template.format(" ".join(items))  # 1 line with options concatenated

        input_ids = self.tokenizer(batch_separate, return_tensors="pt").input_ids.to(torch_device)
        input_ids_shared_prefix = self.tokenizer(batch_shared_prefix, return_tensors="pt").input_ids.to(torch_device)

        mask_shared_prefix = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                ]
            ],
            device=torch_device,
        )

        position_ids = torch.arange(input_ids.shape[1]).tile(input_ids.shape[0], 1).to(torch_device)

        # building custom positions ids based on custom mask
        position_ids_shared_prefix = (mask_shared_prefix.sum(dim=-1) - 1).reshape(1, -1)
        # effectively: position_ids_shared_prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]]).to(device)

        # inverting the mask
        min_dtype = torch.finfo(self.model_dtype).min
        mask_shared_prefix = (mask_shared_prefix.eq(0.0)).to(dtype=self.model_dtype) * min_dtype

        return input_ids, position_ids, input_ids_shared_prefix, mask_shared_prefix, position_ids_shared_prefix

    def test_stacked_causal_mask(self):
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix, attention_mask=mask_shared_prefix, position_ids=position_ids_shared_prefix
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks

        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # 2 forward runs with custom 4D masks
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        outs_1a = self.model.forward(input_1a, attention_mask=mask_1a, position_ids=position_ids_1a)
        past_key_values_a = outs_1a["past_key_values"]

        # Case 1: we pass a 4D attention mask regarding the current sequence length (i.e. [..., seq_len, full_len])
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]
        outs_1b = self.model.forward(
            input_1b,
            attention_mask=mask_1b,
            position_ids=position_ids_1b,
            past_key_values=past_key_values_a,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)

    def test_stacked_causal_mask_static_cache(self):
        """same as above but with StaticCache"""
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(config=self.model.config, max_cache_len=max_cache_len)

        padded_attention_mask = torch.nn.functional.pad(
            input=mask_shared_prefix,
            pad=(0, max_cache_len - mask_shared_prefix.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        # single forward run with 4D custom mask
        logits_shared_prefix = self.model.forward(
            input_ids_shared_prefix,
            attention_mask=padded_attention_mask,
            position_ids=position_ids_shared_prefix,
            cache_position=torch.arange(input_ids_shared_prefix.shape[-1], device=torch_device),
            past_key_values=past_key_values,
        ).logits
        logits_shared_prefix_last = logits_shared_prefix[
            0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1], :
        ]  # last three tokens
        decoded_shared_prefix = [self.tokenizer.decode(t) for t in logits_shared_prefix_last.argmax(dim=-1)]

        self.assertEqual(decoded, decoded_shared_prefix)

    def test_partial_stacked_causal_mask_static_cache(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention masks
        # we pass a 4D attention mask shaped [..., seq_len, full_static_cache_len])
        (
            input_ids,
            position_ids,
            input_ids_shared_prefix,
            mask_shared_prefix,
            position_ids_shared_prefix,
        ) = self.get_test_data()

        # regular batch
        logits = self.model.forward(input_ids, position_ids=position_ids).logits
        logits_last = logits[:, -1, :]  # last tokens in each batch line
        decoded = [self.tokenizer.decode(t) for t in logits_last.argmax(dim=-1)]

        # upgrade the model with StaticCache
        max_cache_len = 16  # note that max_cache_len is greater than the attention_mask.shape[-1]
        past_key_values = StaticCache(config=self.model.config, max_cache_len=max_cache_len)

        # forward run for the first part of input
        part_a = 3  # split point

        input_1a = input_ids_shared_prefix[:, :part_a]
        position_ids_1a = position_ids_shared_prefix[:, :part_a]
        mask_1a = mask_shared_prefix[:, :, :part_a, :part_a]

        padded_mask_1a = torch.nn.functional.pad(
            input=mask_1a,
            pad=(0, max_cache_len - mask_1a.shape[-1]),
            mode="constant",
            value=torch.finfo(self.model_dtype).min,
        )

        _ = self.model.forward(
            input_1a,
            attention_mask=padded_mask_1a,
            position_ids=position_ids_1a,
            cache_position=torch.arange(part_a, device=torch_device),
            past_key_values=past_key_values,
        )

        # forward run for the second part of input
        input_1b = input_ids_shared_prefix[:, part_a:]
        position_ids_1b = position_ids_shared_prefix[:, part_a:]
        mask_1b = mask_shared_prefix[:, :, part_a:, :]

        padded_mask_1b = torch.nn.functional.pad(
            input=mask_1b, pad=(0, max_cache_len - mask_1b.shape[-1]), mode="constant", value=0
        )

        outs_1b = self.model.forward(
            input_1b,
            attention_mask=padded_mask_1b,
            position_ids=position_ids_1b,
            cache_position=torch.arange(
                part_a,
                input_ids_shared_prefix.shape[-1],
                device=torch_device,
            ),
            past_key_values=past_key_values,
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)
