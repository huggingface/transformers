# coding=utf-8
# Copyright 2025 The LG AI Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch EXAONE 4.0 model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Exaone4Config,
    GenerationConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        Exaone4ForCausalLM,
        Exaone4ForQuestionAnswering,
        Exaone4ForSequenceClassification,
        Exaone4ForTokenClassification,
        Exaone4Model,
    )


class Exaone4ModelTester(CausalLMModelTester):
    config_class = Exaone4Config
    if is_torch_available():
        base_model_class = Exaone4Model
        causal_lm_class = Exaone4ForCausalLM
        sequence_class = Exaone4ForSequenceClassification
        token_class = Exaone4ForTokenClassification
        question_answering_class = Exaone4ForQuestionAnswering


@require_torch
class Exaone4ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (
        (
            Exaone4Model,
            Exaone4ForCausalLM,
            Exaone4ForSequenceClassification,
            Exaone4ForQuestionAnswering,
            Exaone4ForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Exaone4Model,
            "question-answering": Exaone4ForQuestionAnswering,
            "text-classification": Exaone4ForSequenceClassification,
            "text-generation": Exaone4ForCausalLM,
            "zero-shot": Exaone4ForSequenceClassification,
            "token-classification": Exaone4ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False  # Broken by attention refactor cc @Cyrilvallez
    model_tester_class = Exaone4ModelTester
    model_split_percents = [0.5, 0.6]

    def setUp(self):
        self.model_tester = Exaone4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Exaone4Config, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support low_memory generation")
    def test_beam_search_low_memory(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip(
        "EXAONE 4.0 has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support."
    )
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("EXAONE 4.0 has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesnt work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class Exaone4IntegrationTest(unittest.TestCase):
    TEST_MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-Instruct"  # dummy model id

    def tearDown(self):
        # TODO (joao): automatic compilation, i.e. compilation when `cache_implementation="static"` is used, leaves
        # some memory allocated in the cache, which means some object is not being released properly. This causes some
        # unoptimal memory usage, e.g. after certain teruff format examples tests src utilssts a 7B model in FP16 no longer fits in a 24GB GPU.
        # Investigate the root cause.
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits(self):
        input_ids = [405, 7584, 79579, 76636, 2907, 94640, 373]
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.float16, attn_implementation="eager"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[13.9380, 12.9951, 12.9442, 10.6576, 11.0901, 12.1466, 9.2482]])
        EXPECTED_SLICE = torch.tensor(
            [
                4.9180,
                11.6406,
                21.1250,
                13.4062,
                20.8438,
                18.0625,
                17.9688,
                18.7812,
                18.0156,
                18.3594,
                18.5000,
                19.1719,
                18.5156,
                19.3438,
                19.5000,
                20.6406,
                19.4844,
                19.2812,
                19.4688,
                20.0156,
                19.8438,
                19.9531,
                19.7188,
                20.5938,
                20.5312,
                20.1250,
                20.4062,
                21.4062,
                21.2344,
                20.7656,
            ]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_logits_bf16(self):
        input_ids = [405, 7584, 79579, 76636, 2907, 94640, 373]
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)
        with torch.no_grad():
            out = model(input_ids).logits.float().cpu()

        EXPECTED_MEAN = torch.tensor([[13.8797, 13.0799, 12.9665, 10.7712, 11.1006, 12.2406, 9.3248]])
        EXPECTED_SLICE = torch.tensor(
            [
                4.8750,
                11.6250,
                21.0000,
                13.3125,
                20.8750,
                18.0000,
                18.0000,
                18.7500,
                18.0000,
                18.3750,
                18.5000,
                19.1250,
                18.5000,
                19.3750,
                19.5000,
                20.6250,
                19.5000,
                19.2500,
                19.5000,
                20.0000,
                19.8750,
                19.8750,
                19.7500,
                20.6250,
                20.5000,
                20.1250,
                20.3750,
                21.3750,
                21.2500,
                20.7500,
            ]
        )

        torch.testing.assert_close(out.mean(-1), EXPECTED_MEAN, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0, 0, :30], EXPECTED_SLICE, atol=1e-4, rtol=1e-4)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_generation(self):
        EXPECTED_TEXT = "Tell me about the Miracle on the Han river.\n\nThe Miracle on the Han River is a story about the miracle of the Korean War Armistice. The story is told by a Korean soldier who is a witness to the armistice negotiations. He is reluctant to tell the story because he does not want to be a hypocrite, but he feels that everyone should know what really happened.\n\nThe Korean War began on June 25, 1950, when North Korean troops invaded South Korea. Soon the United Nations troops, primarily from South Korea, were in support of the United States. The war was still ongoing when North Korean troops stopped their advance"
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.float16, attn_implementation="eager"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=128, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_sdpa
    def test_model_generation_bf16_sdpa(self):
        EXPECTED_TEXT = "Tell me about the Miracle on the Han river.\n\nThe Miracle on the Han River is a story about the miracle of the Korean War Armistice.\n\nThe Korean War broke out in 35 years ago in 1950. The war was the result of the ideological conflict between the communist north and the capitalist south. The war was brought to a halt in 1953. There was to be peace talks but no peace treaty. As a result of the stalemate the Korean people have neither a peace treaty nor a reunification nor a democratization of Korea. The stalemate of 35 years has produced a people of 70 million"
        prompt = "Tell me about the Miracle on the Han river."
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        # greedy generation outputs
        generated_ids = model.generate(input_ids, max_new_tokens=128, temperature=0)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @require_flash_attn
    def test_model_generation_long_flash(self):
        EXPECTED_OUTPUT_TOKEN_IDS = [433, 9055]
        input_ids = [433, 9055] * 2048
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.float16, attn_implementation="flash_attention_2"
        )
        input_ids = torch.tensor([input_ids]).to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=4, temperature=0)
        self.assertEqual(EXPECTED_OUTPUT_TOKEN_IDS, generated_ids[0][-2:].tolist())
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    @require_torch_accelerator
    @require_torch_sdpa
    def test_model_generation_beyond_sliding_window(self):
        EXPECTED_TEXT_COMPLETION = (
            " but I'm not sure if I'm going to be able to see it. I really enjoy the scenery, but I'm not sure if I"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID)
        prompt = "This is a nice place. " * 700 + "I really enjoy the scenery,"
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID, device_map="auto", dtype=torch.float16, attn_implementation="sdpa"
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.model.embed_tokens.weight.device)

        generated_ids = model.generate(input_ids, max_new_tokens=32, temperature=0)
        text = tokenizer.decode(generated_ids[0, -32:], skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
        del model
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_export_static_cache(self):
        if version.parse(torch.__version__) < version.parse("2.4.0"):
            self.skipTest(reason="This test requires torch >= 2.4 to run.")

        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
            convert_and_export_with_cache,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.TEST_MODEL_ID, padding_side="right")
        EXPECTED_TEXT_COMPLETION = [
            "The Deep Learning is 100% free and easy to use.\n\n## How to use Deep Learning?\n\n"
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
        model = Exaone4ForCausalLM.from_pretrained(
            self.TEST_MODEL_ID,
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

        prompt = ["The Deep Learning is "]
        prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_token_ids = prompt_tokens["input_ids"]
        max_new_tokens = max_generation_length - prompt_token_ids.shape[-1]

        # Static Cache + export
        exported_program = convert_and_export_with_cache(model)
        ep_generated_ids = TorchExportableModuleWithStaticCache.generate(
            exported_program=exported_program, prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens
        )
        ep_generated_text = tokenizer.batch_decode(ep_generated_ids, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, ep_generated_text)
