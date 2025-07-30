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
"""Testing suite for the PyTorch OpenaiMoe model."""

import unittest

import pytest
from packaging import version
from parameterized import parameterized
from pytest import mark

from tests.tensor_parallel.test_tensor_parallel import TestTensorParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, OpenaiMoeConfig, is_torch_available, pipeline
from transformers.generation.configuration_utils import GenerationConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    require_flash_attn,
    require_large_cpu_ram,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    require_torch_large_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import (
        OpenAIMoeForCausalLM,
        OpenAIMoeModel,
    )


class OpenaiMoeModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = OpenaiMoeConfig
        base_model_class = OpenAIMoeModel
        causal_lm_class = OpenAIMoeForCausalLM

    pipeline_model_mapping = (
        {
            "feature-extraction": OpenaiMoeModel,
            "text-generation": OpenaiMoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@require_torch
class OpenaiMoeModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (OpenaiMoeModel, OpenaiMoeForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": OpenaiMoeModel,
            "text-generation": OpenaiMoeForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = OpenaiMoeModelTester

    def setUp(self):
        self.model_tester = OpenaiMoeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenaiMoeConfig, hidden_size=37)

    @unittest.skip("Failing because of unique cache (HybridCache)")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @unittest.skip("OpenaiMoe's forcefully disables sdpa due to Sink")
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @unittest.skip("OpenaiMoe's eager attn/sdpa attn outputs are expected to be different")
    def test_eager_matches_sdpa_generate(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip("OpenaiMoe has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("OpenaiMoe has HybridCache which is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass

    @pytest.mark.generate
    @unittest.skip("OpenaiMoe has HybridCache which is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache which is not compatible with dola decoding")
    def test_dola_decoding_sample(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support continue from past kv")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support contrastive generation")
    def test_contrastive_generate_low_memory(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache and doesn't support StaticCache. Though it could, it shouldn't support.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip(
        reason="HybridCache can't be gathered because it is not iterable. Adding a simple iter and dumping `distributed_iterator`"
        " as in Dynamic Cache doesn't work. NOTE: @gante all cache objects would need better compatibility with multi gpu setting"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip("OpenaiMoe has HybridCache which auto-compiles. Compile and FA2 don't work together.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("OpenaiMoe eager/FA2 attention outputs are expected to be different")
    def test_flash_attn_2_equivalence(self):
        pass


@slow
@require_torch_accelerator
class OpenaiMoeIntegrationTest(unittest.TestCase):
    input_text = ["Hello I am doing", "Hi today"]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def load_and_forward(model_id, attn_implementation, input_text, **pretrained_kwargs):
        if not isinstance(attn_implementation, list):
            attn_implementation = [attn_implementation]
        text = []
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **pretrained_kwargs).to(
            torch_device
        )

        for attn in attn_implementation:
            model.set_attn_implementation(attn)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(torch_device)

            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            output_text = tokenizer.batch_decode(output, skip_special_tokens=False)
            text += [output_text]
        return text

    @require_torch_large_accelerator
    @require_read_token
    def test_model_20b_bf16(self):
        model_id = ""
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]
        output_text = self.load_and_forward(
            model_id,
            ["eager", "kernel-community/triton-flash-attn-sink", "ft-hf-o-c/vllm-flash-attn3"],
            self.input_text,
        )
        self.assertEqual(output_text[0], EXPECTED_TEXTS)
        self.assertEqual(output_text[1], EXPECTED_TEXTS)
        self.assertEqual(output_text[2], EXPECTED_TEXTS)

    @require_torch_large_accelerator
    @require_read_token
    def test_model_20b_bf16_use_kernels(self):
        model_id = ""
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]
        output_text = self.load_and_forward(
            model_id,
            ["eager", "kernel-community/triton-flash-attn-sink", "ft-hf-o-c/vllm-flash-attn3"],
            self.input_text,
            use_kenels=True,
        )
        self.assertEqual(output_text[0], EXPECTED_TEXTS)
        self.assertEqual(output_text[1], EXPECTED_TEXTS)
        self.assertEqual(output_text[2], EXPECTED_TEXTS)

    @require_torch_large_accelerator
    @require_read_token
    def test_model_120b_bf16_use_kernels(self):
        model_id = ""
        EXPECTED_TEXTS = [
            "<bos>Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
            "<pad><pad><bos>Hi today I'm going to be talking about the history of the United States. The United States of America",
        ]
        output_text = self.load_and_forward(
            model_id,
            ["eager", "kernel-community/triton-flash-attn-sink", "ft-hf-o-c/vllm-flash-attn3"],
            self.input_text,
            use_kenels=True,
        )
        self.assertEqual(output_text[0], EXPECTED_TEXTS)
        self.assertEqual(output_text[1], EXPECTED_TEXTS)
        self.assertEqual(output_text[2], EXPECTED_TEXTS)

class OpenAIMoeTPTest(TestTensorParallel):


