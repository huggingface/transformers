# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llama4 model."""

import unittest

from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Llama4Config,
        Llama4ForConditionalGeneration,
        Llama4Processor,
    )
    from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding


class Llama4VisionText2TextModelTester(VLMModelTester):
    config_class = Llama4Config
    # Llama 4 has no base multimodal model: the vision tower, the projector and the language model sit directly
    # in `Llama4ForConditionalGeneration`
    base_model_class = None
    conditional_generation_class = Llama4ForConditionalGeneration

    small_config_overrides = {
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        # one dense layer and one MoE layer
        "interleave_moe_layer_step": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 1,
        "intermediate_size_mlp": 48,
        "no_rope_layers": [1, 0],
        # longer than every test sequence, and keeps the static cache of the chunked-attention layers small
        "attention_chunk_size": 64,
        "image_size": 16,
        "patch_size": 4,
        # a (16 // 4) ** 2 patch grid, pixel-shuffled with ratio 0.5, gives 4 image tokens
        "num_image_tokens": 4,
        # the pixel shuffle turns `hidden_size` channels into `hidden_size / 0.5**2`, the input of the vision adapter
        "vision_intermediate_size": 128,
        "vision_output_dim": 32,
        "projector_input_dim": 32,
        "projector_output_dim": 32,
    }

    @property
    def pipeline_model_mapping(self):
        return {"image-text-to-text": self.conditional_generation_class}


@require_torch
class Llama4ForConditionalGenerationModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Llama4VisionText2TextModelTester

    @unittest.skip(
        "Llama 4 has no base multimodal model: the vision tower, the projector and the language model are direct "
        "children of `Llama4ForConditionalGeneration`"
    )
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(
        "The plan keeps entries that match no parameter (`lm_head` and `language_model.layers.*`, next to the valid "
        "`language_model.lm_head` and `language_model.model.layers.*`), because neither `base_model_prefix` names a "
        "submodule; changing them changes checkpoint loading"
    )
    def test_tp_plan_matches_params(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip(
        "The test runs the `MODEL_MAPPING` class as a base model, which for Llama 4 is "
        "`Llama4ForConditionalGeneration`, and its output has no `last_hidden_state`"
    )
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def test_model_rope_scaling_frequencies(self):
        """
        Overwritten: Llama 4 implements RoPE in the complex domain, as opposed to in the real domain with `sin` and
        `cos`. Nevertheless, the checks are the same as in the original test.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        text_config = config.get_text_config()
        scaling_factor = 10
        short_input_length = 10
        long_input_length = int(text_config.max_position_embeddings * 1.5)

        # Inputs
        # `x` is used exclusively to get the dtype and the device
        x = torch.randn(1, dtype=torch.float32, device=torch_device)
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device).unsqueeze(0)

        # Sanity check original RoPE
        text_config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        original_rope = Llama4TextRotaryEmbedding(config=text_config).to(torch_device)
        original_freqs_cis_short = original_rope(x, position_ids_short)
        original_freqs_cis_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_freqs_cis_short, original_freqs_cis_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        text_config.rope_parameters = {"rope_type": "linear", "rope_theta": 10000.0, "factor": scaling_factor}
        linear_scaling_rope = Llama4TextRotaryEmbedding(config=text_config).to(torch_device)
        linear_freqs_cis_short = linear_scaling_rope(x, position_ids_short)
        linear_freqs_cis_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_freqs_cis_short, linear_freqs_cis_long[:, :short_input_length, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        text_config.rope_parameters = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": scaling_factor}
        ntk_scaling_rope = Llama4TextRotaryEmbedding(config=text_config).to(torch_device)
        ntk_freqs_cis_short = ntk_scaling_rope(x, position_ids_short)
        ntk_freqs_cis_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_freqs_cis_long, original_freqs_cis_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        text_config.rope_parameters = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": scaling_factor}
        yarn_scaling_rope = Llama4TextRotaryEmbedding(config=text_config).to(torch_device)
        yarn_freqs_cis_short = yarn_scaling_rope(x, position_ids_short)
        yarn_freqs_cis_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_freqs_cis_short, yarn_freqs_cis_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_short, original_freqs_cis_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_freqs_cis_long, original_freqs_cis_long)

    @unittest.skip("`PreTrainedModel.get_compiled_call` never compiles Llama 4, so generation does not auto-compile")
    def test_generate_compilation_all_outputs(self):
        pass

    @unittest.skip("`PreTrainedModel.get_compiled_call` never compiles Llama 4, so generation does not auto-compile")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip("`PreTrainedModel.get_compiled_call` never compiles Llama 4, so generation does not auto-compile")
    def test_static_cache_no_recompile_with_smaller_length(self):
        pass


@slow
@require_torch_large_accelerator
class Llama4IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "meta-llama/Llama-4-Scout-17B-16E"

    @classmethod
    def setUpClass(cls):
        cls.model = Llama4ForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-4-Scout-17B-16E",
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="eager",
        )

    def setUp(self):
        super().setUp()
        self.processor = Llama4Processor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages_1 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        self.messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
                        ),
                    },
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
                        ),
                    },
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

    def test_model_17b_16e_fp32(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach with a blue sky and a body of water in the background. The cow is brown with a white face'],
                ("cuda", None): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        inputs = self.processor.apply_chat_template(
            self.messages_1, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device=torch_device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        print(output_text)
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_17b_16e_batch(self):
        inputs = self.processor.apply_chat_template(
            [self.messages_1, self.messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(device=torch_device, dtype=torch.float32)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white',
            'system\n\nYou are a helpful assistant.user\n\nAre these images identical?assistant\n\nNo, these images are not identical. The first image shows a cow standing on a beach with a blue sky and a white cloud in the background.'
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)
