# Copyright 2023 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Mistral model."""

import gc
import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import AutoTokenizer, DynamicCache, MistralConfig, is_torch_available, set_seed
from transformers.cache_utils import DynamicSlidingWindowLayer
from transformers.testing_utils import (
    DeviceProperties,
    Expectations,
    backend_empty_cache,
    cleanup,
    get_device_properties,
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import (
        MistralForCausalLM,
        MistralForQuestionAnswering,
        MistralForSequenceClassification,
        MistralForTokenClassification,
        MistralModel,
    )

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class MistralModelTester(CausalLMModelTester):
    config_class = MistralConfig
    if is_torch_available():
        base_model_class = MistralModel
        causal_lm_class = MistralForCausalLM
        sequence_class = MistralForSequenceClassification
        token_class = MistralForTokenClassification
        question_answering_class = MistralForQuestionAnswering


@require_torch
class MistralModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            MistralModel,
            MistralForCausalLM,
            MistralForSequenceClassification,
            MistralForTokenClassification,
            MistralForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MistralModel,
            "text-classification": MistralForSequenceClassification,
            "token-classification": MistralForTokenClassification,
            "text-generation": MistralForCausalLM,
            "question-answering": MistralForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    model_tester_class = MistralModelTester

    # TODO (ydshieh): Check this. See https://app.circleci.com/pipelines/github/huggingface/transformers/79245/workflows/9490ef58-79c2-410d-8f51-e3495156cf9c/jobs/1012146
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True


@require_torch_accelerator
@require_read_token
class MistralIntegrationTest(unittest.TestCase):
    # This variable is used to determine which accelerator are we using for our runners (e.g. A10 or T4)
    # Depending on the hardware we get different logits / generations
    device_properties: DeviceProperties = (None, None, None)

    @classmethod
    def setUpClass(cls):
        cls.device_properties = get_device_properties()

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_7b_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", dtype=torch.float16)
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()
        # Expected mean on dim = -1
        EXPECTED_MEAN = torch.tensor([[-2.5548, -2.5737, -3.0600, -2.5906, -2.8478, -2.8118, -2.9325, -2.7694]])
        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)

        # ("cuda", 8) for A100/A10, and ("cuda", 7) 7 for T4.
        # considering differences in hardware processing and potential deviations in output.
        # fmt: off
        EXPECTED_SLICES = Expectations(
            {
                ("cuda", 7): torch.tensor([-5.8828, -5.8633, -0.1042, -4.7266, -5.8828, -5.8789, -5.8789, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -1.0801,  1.7598, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828, -5.8828]),
                ("cuda", 8): torch.tensor([-5.8711, -5.8555, -0.1050, -4.7148, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -1.0781, 1.7568, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711, -5.8711]),
                ("rocm", 9): torch.tensor([-5.8750, -5.8594, -0.1047, -4.7188, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -1.0781,  1.7578, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750, -5.8750]),
            }
        )
        # fmt: on
        expected_slice = EXPECTED_SLICES.get_expectation()

        torch.testing.assert_close(out[0, 0, :30], expected_slice, atol=1e-4, rtol=1e-4)

    @slow
    @require_bitsandbytes
    def test_model_7b_generation(self):
        EXPECTED_TEXT_COMPLETION = "My favourite condiment is 100% ketchup. I’m not a fan of mustard, mayo,"

        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        model = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", device_map={"": torch_device}, load_in_4bit=True
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    # TODO joao, manuel: remove this in v4.62.0
    @slow
    def test_model_7b_dola_generation(self):
        # ground truth text generated with dola_layers="low", repetition_penalty=1.2
        EXPECTED_TEXT_COMPLETION = (
            """My favourite condiment is 100% ketchup. I love it on everything, and I’m not ash"""
        )
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", dtype=torch.float16)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0,
            dola_layers="low",
            repetition_penalty=1.2,
            trust_remote_code=True,
            custom_generate="transformers-community/dola",
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @require_flash_attn
    @require_bitsandbytes
    @slow
    @pytest.mark.flash_attn_test
    def test_model_7b_long_prompt(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map={"": torch_device},
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

    @slow
    def test_model_7b_long_prompt_sdpa(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [306, 338]
        # An input with 4097 tokens that is above the size of the sliding window
        input_ids = [1] + [306, 338] * 2048
        model = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", device_map="auto", attn_implementation="sdpa", dtype=torch.float16
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        # Assisted generation
        assistant_model = model
        assistant_model.generation_config.num_assistant_tokens = 2
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"
        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())

        del assistant_model

        backend_empty_cache(torch_device)
        gc.collect()

        EXPECTED_TEXT_COMPLETION = """My favourite condiment is 100% ketchup. I love it on everything. I’m not a big"""
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    def test_speculative_generation(self):
        EXPECTED_TEXT_COMPLETION = "My favourite condiment is 100% Sriracha. I love it on everything. I have it on my"
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", dtype=torch.float16)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        set_seed(0)
        generated_ids = model.generate(
            input_ids, max_new_tokens=20, do_sample=True, temperature=0.3, assistant_model=model
        )
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @pytest.mark.torch_compile_test
    @slow
    def test_compile_static_cache(self):
        # `torch==2.2` will throw an error on this test (as in other compilation tests), but torch==2.1.2 and torch>2.2
        # work as intended. See https://github.com/pytorch/pytorch/issues/121943
        if version.parse(torch.__version__) < version.parse("2.3.0"):
            self.skipTest(reason="This test requires torch >= 2.3 to run.")

        if self.device_properties[0] == "cuda" and self.device_properties[1] == 7:
            self.skipTest(reason="This test is failing (`torch.compile` fails) on Nvidia T4 GPU.")

        NUM_TOKENS_TO_GENERATE = 40
        EXPECTED_TEXT_COMPLETION = [
            "My favourite condiment is 100% ketchup. I love it on everything. "
            "I’m not a big fan of mustard, mayo, or relish. I’m not a fan of pickles"
        ]

        prompts = ["My favourite condiment is "]
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", device_map=torch_device, dtype=torch.float16
        )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        # Dynamic Cache
        generated_ids = model.generate(**inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False)
        dynamic_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, dynamic_text)

        # Static Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Sliding Window Cache
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="sliding_window"
        )
        static_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_text)

        # Static Cache + compile
        forward_function = model.__call__
        model.__call__ = torch.compile(forward_function, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="static"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)

        # Sliding Window Cache + compile
        torch._dynamo.reset()
        model.__call__ = torch.compile(forward_function, mode="reduce-overhead", fullgraph=True)
        generated_ids = model.generate(
            **inputs, max_new_tokens=NUM_TOKENS_TO_GENERATE, do_sample=False, cache_implementation="sliding_window"
        )
        static_compiled_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, static_compiled_text)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("flex_attention",), ("eager",)])
    @require_flash_attn
    @slow
    def test_generation_beyond_sliding_window_dynamic(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non-trivial as Mistral will use
        a DynamicCache with only sliding layers."""

        # Impossible to test it with this model (even with < 100 tokens), probably due to the compilation of a large model.
        if attn_implementation == "flex_attention":
            self.skipTest(
                reason="`flex_attention` gives `torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_0 Required: 147456 Hardware limit:101376 Reducing block sizes or `num_stages` may help.`"
            )

        model_id = "mistralai/Mistral-7B-v0.1"
        EXPECTED_COMPLETIONS = [
            "scenery, scenery, scenery, scenery, scenery,",
            ", green, yellow, orange, purple, pink, brown, black, white, gray, silver",
        ]

        input_text = [
            "This is a nice place. " * 682 + "I really enjoy the scenery,",  # This has 4101 tokens, 15 more than 4096
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]

        if attn_implementation == "eager":
            input_text = input_text[:1]

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = MistralForCausalLM.from_pretrained(
            model_id, attn_implementation=attn_implementation, device_map=torch_device, dtype=torch.float16
        )

        # Make sure prefill is larger than sliding window
        batch_size, input_size = inputs.input_ids.shape
        self.assertTrue(input_size > model.config.sliding_window)

        # Should already be Dynamic by default, but let's make sure!
        out = model.generate(**inputs, max_new_tokens=20, cache_implementation="dynamic", return_dict_in_generate=True)
        output_text = tokenizer.batch_decode(out.sequences[:batch_size, input_size:])

        self.assertEqual(output_text, EXPECTED_COMPLETIONS[:batch_size])

        # Let's check that the dynamic cache has hybrid layers!
        dynamic_cache = out.past_key_values
        self.assertTrue(isinstance(dynamic_cache, DynamicCache))
        for layer in dynamic_cache.layers:
            self.assertTrue(isinstance(layer, DynamicSlidingWindowLayer))
            self.assertEqual(layer.keys.shape[-2], model.config.sliding_window - 1)


@slow
@require_torch_accelerator
class Mask4DTestHard(unittest.TestCase):
    model_name = "mistralai/Mistral-7B-v0.1"
    model = None
    model_dtype = None

    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        if cls.model_dtype is None:
            cls.model_dtype = torch.float16
        if cls.model is None:
            cls.model = MistralForCausalLM.from_pretrained(cls.model_name, dtype=cls.model_dtype).to(torch_device)

    @classmethod
    def tearDownClass(cls):
        del cls.model_dtype
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

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
            input_1b, attention_mask=mask_1b, position_ids=position_ids_1b, past_key_values=past_key_values_a
        )
        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[
                0, torch.where(position_ids_shared_prefix == position_ids_shared_prefix.max())[1] - part_a
            ]
        ]
        self.assertEqual(decoded, decoded_1b)
