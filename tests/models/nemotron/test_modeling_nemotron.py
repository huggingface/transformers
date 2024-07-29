# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Testing suite for the PyTorch Nemotron model."""

import gc
import tempfile
import unittest

import pytest
from packaging import version
from parameterized import parameterized

from transformers import NemotronConfig, StaticCache, is_torch_available, set_seed
from transformers.testing_utils import (
    require_bitsandbytes,
    require_flash_attn,
    require_read_token,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin

from ...models.gemma.test_modeling_gemma import GemmaModelTest, GemmaModelTester

if is_torch_available():
    import torch

    from transformers import (
        NemotronForCausalLM,
        NemotronForQuestionAnswering,
        NemotronForSequenceClassification,
        NemotronForTokenClassification,
        NemotronModel,
        AutoTokenizer,
    )
    from transformers.models.nemotron.modeling_nemotron import (
        NemotronLinearScalingRotaryEmbedding,
        NemotronRotaryEmbedding,
    )


class NemotronModelTester(GemmaModelTester):
    if is_torch_available():
        config_class = NemotronConfig
        model_class = NemotronModel
        for_causal_lm_class = NemotronForCausalLM
        for_sequence_class = NemotronForSequenceClassification
        for_token_class = NemotronForTokenClassification


@require_torch
class NemotronModelTest(GemmaModelTest):
    all_model_classes = (
        (
            NemotronModel,
            NemotronForCausalLM,
            NemotronForSequenceClassification,
            NemotronForQuestionAnswering,
            NemotronForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (NemotronForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": NemotronModel,
            "text-classification": NemotronForSequenceClassification,
            "text-generation": NemotronForCausalLM,
            "zero-shot": NemotronForSequenceClassification,
            "question-answering": NemotronForQuestionAnswering,
            "token-classification": NemotronForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    test_headmasking = False
    test_pruning = False
    fx_compatible = False

    # used in `test_torch_compile`
    _torch_compile_test_ckpt = "nvidia/nemotron3-7b-hf"

    def setUp(self):
        self.model_tester = NemotronModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_nemotron_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = NemotronForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_nemotron_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = NemotronForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_nemotron_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = NemotronForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_nemotron_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = NemotronForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @require_torch_sdpa
    @slow
    @unittest.skip(reason="Due to custom causal mask, there is a slightly too big difference between eager and sdpa in bfloat16.")
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        pass

    @require_torch_sdpa
    @slow
    @unittest.skip(reason="Due to custom causal mask, there is a slightly too big difference between eager and sdpa in bfloat16.")
    @parameterized.expand([("float32",)])
    def test_eager_matches_sdpa_inference_2(self, torch_dtype: str):
        pass

    @unittest.skip("Eager and SDPA do not produce the same outputs, thus this test fails")
    def test_model_outputs_equivalence(self, **kwargs):
        pass
    
    @parameterized.expand([("linear",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = NemotronModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 10.0}
        scaled_model = NemotronModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state


        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        original_rope = NemotronRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        linear_scaling_rope = NemotronLinearScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

    @require_flash_attn
    @require_torch_gpu
    @slow
    def test_use_flash_attention_2_true(self):
        """
        NOTE: this is the only test testing that the legacy `use_flash_attention=2` argument still works as intended.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes: 
            with tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(config)
                model.save_pretrained(tmp_dir)

                new_model = NemotronForCausalLM.from_pretrained(
                    tmp_dir, use_flash_attention_2=True, torch_dtype=torch.float16
                ).to("cuda")

                self.assertTrue(new_model.config._attn_implementation == "flash_attention_2")

                has_flash = False
                for name, submodule in new_model.named_modules():
                    if "FlashAttention" in submodule.__class__.__name__:
                        has_flash = True
                        break
                if not has_flash:
                    raise ValueError("The flash model should have flash attention layers")

    @require_torch_sdpa
    @slow
    def test_eager_matches_sdpa_generate(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        max_new_tokens = 30

        test_model = "thhaus/nemotron3-8b"
        tokenizer = AutoTokenizer.from_pretrained(test_model)

        model_sdpa = NemotronForCausalLM.from_pretrained(
            test_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(torch_device)

        self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        model_eager = NemotronForCausalLM.from_pretrained(
            test_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(torch_device)

        self.assertTrue(model_eager.config._attn_implementation == "eager")

        for name, submodule in model_eager.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                raise ValueError("The eager model should not have SDPA attention layers")

        has_sdpa = False
        for name, submodule in model_sdpa.named_modules():
            if "SdpaAttention" in submodule.__class__.__name__:
                has_sdpa = True
                break
        if not has_sdpa:
            raise ValueError("The SDPA model should have SDPA attention layers")

        texts = [
            "hi here's a longer context, getting longer and",
            "Hello this is a very long sentence my friend, very long for real",
            "Today I am in Paris and",
        ]

        for padding_side in ["left", "right"]:
            tokenizer.padding_side = padding_side
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(torch_device)

            res_eager = model_eager.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            res_sdpa = model_sdpa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            with self.subTest(f"{padding_side}"):
                torch.testing.assert_close(
                    res_eager,
                    res_sdpa,
                    msg=f"\n{tokenizer.batch_decode(res_eager)} \nvs\n{tokenizer.batch_decode(res_sdpa)}",
                )
    
    @require_flash_attn
    @require_torch_gpu
    @require_bitsandbytes
    @pytest.mark.flash_attn_test
    @require_read_token
    @slow
    def test_flash_attn_2_generate_padding_right(self):
        """
        Overwritting the common test as the test is flaky on tiny models
        """
        model = NemotronForCausalLM.from_pretrained(
            "thhaus/nemotron3-8b",
            torch_dtype=torch.float16, device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained("thhaus/nemotron3-8b")

        texts = ["hi", "Hello this is a very long sentence"]

        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(0)

        output_native = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_native = tokenizer.batch_decode(output_native)

        model = NemotronForCausalLM.from_pretrained(
            "thhaus/nemotron3-8b", torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )

        output_fa_2 = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        output_fa_2 = tokenizer.batch_decode(output_fa_2)

        self.assertListEqual(output_native, output_fa_2)


@require_torch_gpu
class NemotronIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    @slow
    @require_read_token
    def test_nemotron_8b_generation_sdpa(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)
        
        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_eager(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)
        
        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

    @slow
    @require_read_token
    def test_nemotron_8b_generation_fa2(self):
        text = ["What is the largest planet in solar system?"]
        EXPECTED_TEXT = [
            "What is the largest planet in solar system?\nAnswer: Jupiter\n\nWhat is the answer",
        ]
        model_id = "thhaus/nemotron3-8b"
        model = NemotronForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)
        
        output = model.generate(**inputs, do_sample=False)
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(EXPECTED_TEXT, output_text)

