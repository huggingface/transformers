# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma4 model."""

import logging
import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4Config,
    Gemma4TextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    is_flash_attn_2_available,
    require_deterministic_for_xpu,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4Processor,
        Gemma4TextModel,
    )
    from transformers.pytorch_utils import is_torch_greater_or_equal


class Gemma4TextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Gemma4TextConfig
        base_model_class = Gemma4TextModel
        causal_lm_class = Gemma4ForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = 4  # override to correctly test sharing cache pattern
        self.num_kv_shared_layers = 2  # important to override
        self.layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]  # similarly we want to test sharing on both types
        self.global_head_dim = self.head_dim  # gemma4 use a different head_dim for full and sliding layers

        # To make model small
        self.vocab_size_per_layer_input = 99
        self.hidden_size_per_layer_input = 16

        # To activate moe blocks
        self.enable_moe_block = True
        self.moe_intermediate_size = 16
        self.top_k_experts = 2

        # Test if bidirectional image mask path works
        self.use_bidirectional_attention = "vision"


@require_torch
class Gemma4TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Gemma4TextModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Gemma4ForCausalLM if is_torch_available() else None

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4 uses different rope per layer type, which is not compatible with this test")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("Gemma4 uses different rope per layer type, which is not compatible with this test")
    def test_model_rope_scaling_from_config(self):
        pass

    @unittest.skip(
        "Gemma4 cannot use random inputs_embeds, as it needs to reverse them when input_ids is not provided"
    )
    def test_generate_from_random_inputs_embeds(self):
        pass

    def test_use_cache_false_with_kv_sharing(self):
        """Regression test: use_cache=False must produce the same logits as use_cache=True.

        Gemma4 uses KV sharing (num_kv_shared_layers) where later layers reuse K/V from earlier
        layers via the cache object. When use_cache=False the cache was not created, breaking the
        sharing mechanism and causing receiver layers to use keys as values (garbage logits).
        See https://github.com/huggingface/transformers/issues/45242
        """
        config = self.model_tester.get_config()
        config.attention_k_eq_v = True
        config.num_global_key_value_heads = config.num_key_value_heads
        model = Gemma4ForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor([1, 16], config.vocab_size).to(torch_device)

        with torch.no_grad():
            out_cached = model(input_ids, use_cache=True)
            out_uncached = model(input_ids, use_cache=False)

        torch.testing.assert_close(out_cached.logits, out_uncached.logits, atol=1e-4, rtol=1e-4)
        self.assertIsNone(out_uncached.past_key_values, "past_key_values should be None when use_cache=False")

    @unittest.skip(
        "Flaky on CI, but not locally on Mac. If model is set to fp32 instead of bf16, not flaky anymore."
        "TODO Cyril: investigate where the loss of precision between bf16 and fp32 comes from."
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass


class Gemma4Audio2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_id=4,
        boi_token_id=5,
        eoi_token_id=6,
        audio_token_id=7,
        boa_token_id=8,
        eoa_token_index=9,
        video_token_id=10,
        seq_length=50,
        audio_seq_length=96,
        audio_num_channels=16,
        is_training=True,
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_act": "silu",
            "subsampling_conv_channels": [16, 8],
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 5,
            "attention_context_right": 0,
            "output_proj_dims": 32,
            # Clipped linears register inf/-inf buffers which cause NaN in test_torch_save_load's
            # comparison logic (inf - inf = NaN). Disable for testing.
            "use_clipped_linears": False,
        },
    ):
        self.parent = parent
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.audio_token_id = audio_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        self.video_token_id = video_token_id
        self.llm_tester = Gemma4TextModelTester(self.parent)
        self.llm_tester.use_bidirectional_attention = None
        self.text_config = self.llm_tester.get_config()
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.audio_seq_length = audio_seq_length
        self.audio_num_channels = audio_num_channels
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Gemma4Config(
            text_config=self.text_config,
            vision_config=None,
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            audio_token_id=self.audio_token_id,
            boa_token_id=self.boa_token_id,
            eoa_token_index=self.eoa_token_index,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.audio_seq_length, self.audio_num_channels])
        input_features_mask = torch.ones(self.batch_size, self.audio_seq_length, dtype=torch.bool)
        config = self.get_config()
        return config, input_features, input_features_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_features, input_features_mask = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id

        # The audio encoder produces audio_seq_length / 4 tokens per audio sample after subsampling.
        # We need that many audio placeholder tokens per sequence in input_ids.
        num_audio_tokens = self.audio_seq_length // 4
        input_ids[:, :num_audio_tokens] = config.audio_token_id

        inputs_dict = {
            "input_features": input_features,
            "input_features_mask": input_features_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Gemma4Audio2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma4Model, Gemma4ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma4ForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = Gemma4Audio2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma4Config, hidden_size=37)

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    @parameterized.expand([True, False, None])
    def test_get_image_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    @parameterized.expand([True, False, None])
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4 needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass


class Gemma4Vision2TextModelTester:
    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_id=4,
        video_token_id=7,
        audio_token_id=8,
        boi_token_id=5,
        eoi_token_id=6,
        seq_length=25,
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        # `image_token_id` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.llm_tester = Gemma4TextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Gemma4Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            mm_tokens_per_image=self.mm_tokens_per_image,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        config.vision_config.pooling_kernel_size = 2

        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["image_size"],
                patch_size * patch_size * self.vision_config["num_channels"],
            ]
        )
        # (num_images, max_num_patches, 2) for height/width positions. Let it be all ones for testign
        pixel_position_ids = torch.ones(self.vision_config["image_size"], device=torch_device, dtype=torch.long)
        pixel_position_ids = pixel_position_ids[None, :, None].repeat(self.batch_size, 1, 2)

        return config, pixel_values, pixel_position_ids

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_position_ids = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id
        input_ids[:, :1] = config.image_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == config.image_token_id] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_position_ids": pixel_position_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Gemma4Vision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma4Model, Gemma4ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma4ForConditionalGeneration,) if is_torch_available() else ()
    additional_model_inputs = ["mm_token_type_ids"]

    def setUp(self):
        self.model_tester = Gemma4Vision2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma4Config, hidden_size=37)

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no audios in input dict")
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip("The tester has no audios in input dict")
    @parameterized.expand([True, False, None])
    def test_get_audio_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    def test_get_video_features_attentions(self):
        pass

    @unittest.skip("The tester has no videos in input dict")
    @parameterized.expand([True, False, None])
    def test_get_video_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4 needs correct embeddings for per-layer-input computation, random won't work!")
    def test_generate_from_random_inputs_embeds(self):
        pass


@unittest.skip("Integration Tests are not up-to-date yet! TODO Cyril: update me pretty pretty please!")
@slow
@require_torch_accelerator
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = Gemma4Processor.from_pretrained("google/gemma-4-e2b-it", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_deterministic_for_xpu
    def test_model_4b_bf16(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach with turquoise water in the background. It looks like a lovely,'],
                ("cuda", (8, 0)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like'],
                ("cuda", (8, 6)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear blue water and a blue sky in the background. It looks like'],
                ("rocm", (9, 4)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with turquoise water and a blue sky in the background. It looks like a'],
                ("rocm", (9, 5)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant coastline in the background. It looks'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    @require_deterministic_for_xpu
    def test_model_4b_batch(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. It looks like a very sunny and',
                        'user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. They depict very different scenes:\n\n*   **Image 1** shows a cow standing on a beach.',
                    ],
                ("cuda", (8,0)):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like',
                        "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a brown"
                        ],
                ("cuda", (8,6)):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear blue water and a blue sky in the background. It looks like',
                        "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a brown"
                    ],
                ("rocm", (9, 4)):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown cow standing on a sandy beach with clear turquoise water and a blue sky in the background. It looks like',
                        "user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. \n\nHere's a breakdown of the differences:\n\n*   **Image 1:** Shows a cow"
                    ],
                ("rocm", (9, 5)):
                    [
                        'user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nCertainly! \n\nThe image shows a brown and white cow standing on a sandy beach next to a turquoise ocean. There are some clouds in the blue',
                        'user\nYou are a helpful assistant.\n\n\n\n\n\n\n\n\n\nAre these images identical?\nmodel\nNo, these images are not identical. They depict very different scenes. \n\n*   **Image 1** shows a cow standing on a beach',
                    ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    def test_model_4b_crops(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        crop_config = {
            "images_kwargs": {
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 448,
                "pan_and_scan_min_crop_size": 32,
                "pan_and_scan_min_ratio_to_activate": 0.3,
            }
        }

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            **crop_config,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_NUM_IMAGES = 3  # one for the origin image and two crops of images
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.'],
                ("cuda", 7): [],
                ("cuda", (8, 6)): ["user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a clear blue sky with some white clouds above."],
                ("cuda", (8, 0)): ["user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a blue sky with some white clouds in the background"],
                ("rocm", (9, 4)): ["user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a bright blue sky with some white clouds in the"],
                ("rocm", (9, 5)): ["user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a blue sky with some white clouds in the background"]
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        print(f"Generated text: {output_text}")
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    @require_deterministic_for_xpu
    def test_model_4b_batch_crops(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)
        crop_config = {
            "images_kwargs": {
                "do_pan_and_scan": True,
                "pan_and_scan_max_num_crops": 448,
                "pan_and_scan_min_crop_size": 32,
                "pan_and_scan_min_ratio_to_activate": 0.3,
            }
        }
        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                    },
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
            **crop_config,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_NUM_IMAGES = 9  # 3 * (one for the origin image and two crops of images) = 9
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): [
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.',
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a',
                ],
                ("cuda", 7): [],
                ("cuda", (8,0)): [
                    "user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a blue sky with some white clouds in the background",
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a'
                    ],
                ("cuda", (8, 6)): [
                    "user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a bright blue sky with some white clouds in the",
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a'
                ],
                ("rocm", (9, 4)) : [
                    "user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There's a bright blue sky with some white clouds in the",
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a'
                    ],
                ("rocm", (9, 5)) : [
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown cow standing on a sandy beach next to a turquoise ocean. There are clouds in the blue sky above.',
                    'user\nYou are a helpful assistant.\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nHere is the original image \n\n\n\n and here are some crops to help you see better \n\n\n\n \n\n\n\nAre these images identical?\nmodel\nNo, the images are not identical. \n\nThe first image shows a cow on a beach, while the second image shows a street scene with a',
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(len(inputs["pixel_values"]), EXPECTED_NUM_IMAGES)
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_large_accelerator
    def test_model_4b_multiimage(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image!\n\nHere's a description of the scene:\n\n*   **Chinese Arch"],
                ("cuda", 7): [],
                ("cuda", (8, 0)): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Overall Scene:**\n\nIt looks like a street scene in a vibrant,"],
                ("cuda", (8, 6)): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Overall Scene:**\n\nIt appears to be a street scene in a city"],
                ("rocm", (9, 4)): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Overall Scene:**\n\nIt appears to be a street scene in a vibrant"],
                ("rocm", (9, 5)): ["user\nYou are a helpful assistant.\n\n\n\n\n\nWhat do you see here?\nmodel\nOkay, let's break down what I see in this image:\n\n**Main Features:**\n\n*   **Chinese Archway:** The most prominent"],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_1b_text_only(self):
        model_id = "google/gemma-3-1b-it"

        model = Gemma4ForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        inputs = tokenizer("Write a poem about Machine Learning.", return_tensors="pt").to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning'],
                ("cuda", 7): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a silent stream,\nInto the neural net, a waking dream.\nAlgorithms hum, a coded grace,\n'],
                ("cuda", 8): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a silent stream,\nInto the neural net, a waking dream.\nAlgorithms hum, a coded grace,\n'],
                ("rocm", (9, 4)): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a silent stream,\nInto the neural net, a waking dream.\nAlgorithms hum, a coded grace,\n'],
                ("rocm", (9, 5)): ['Write a poem about Machine Learning.\n\n---\n\nThe data flows, a river deep,\nWith patterns hidden, secrets sleep.\nA neural net, a watchful eye,\nLearning'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    # TODO: raushan FA2 generates gibberish for no reason, check later
    @require_flash_attn
    @require_torch_large_accelerator
    @pytest.mark.flash_attn_test
    def test_model_4b_flash_attn(self):
        model_id = "google/gemma-4-e2b-it"

        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to(torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        # cache_implementation="hybrid" an in the original transformers implementation
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, cache_implementation="hybrid")
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant island in the background. It looks like a sunny day'],
                ("cuda", 7): [],
                ("cuda", 8): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with turquoise water and a distant island in the background. It looks like a sunny day'],
                ("rocm", (9, 5)): ['user\nYou are a helpful assistant.\n\n\n\n\n\nWhat is shown in this image?\nmodel\nThe image shows a brown and white cow standing on a sandy beach with a turquoise ocean and a distant island in the background. It looks like a sunny'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @parameterized.expand([("flash_attention_2",), ("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. This is non trivial as
        we need to correctly slice the attention mask in all cases (because we use a hybrid cache).
        Outputs for every attention functions should be coherent and identical.
        """
        model_id = "google/gemma-3-1b-it"

        if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
            self.skipTest("FlashAttention2 is required for this test.")

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

        out = model.generate(**inputs, max_new_tokens=20, do_sample=False, cache_implementation="static")[
            :, input_size:
        ]
        output_text = tokenizer.batch_decode(out)

        EXPECTED_COMPLETIONS = [" and I'm going to take a walk.\n\nI really enjoy the scenery, and I'", ", green, yellow, orange, purple, brown, black, white, gray.\n\nI'"]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_COMPLETIONS)

    @pytest.mark.torch_export_test
    def test_export_text_only_with_hybrid_cache(self):
        if not is_torch_greater_or_equal("2.6.0"):
            self.skipTest(reason="This test requires torch >= 2.6 to run.")

        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        model_id = "google/gemma-3-1b-it"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        self.assertEqual(model.config.cache_implementation, "hybrid")

        # Export + hybrid cache
        model.eval()
        exportable_module = TorchExportableModuleForDecoderOnlyLM(model, batch_size=1, max_cache_len=1024)
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], dtype=torch.long, device=model.device),
        )
        logging.info(f"\nExported program: {exported_program}")

        # Test generation with the exported model
        prompt = "What is the capital of France?"
        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate
        )
        logging.info(f"\nExport generated texts: '{export_generated_text}'")

        input_text = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            eager_outputs = model.generate(
                **input_text,
                max_new_tokens=max_new_tokens_to_generate,
                do_sample=False,  # Use greedy decoding to match the exported model
                cache_implementation="hybrid",
            )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        logging.info(f"\nEager generated texts: '{eager_generated_text}'")

        self.assertEqual(export_generated_text, eager_generated_text)

    def test_dynamic_sliding_window_is_default(self):
        """
        Test that the dynamic sliding window cache (added in #40039) is the default cache implementation for Gemma4
        models, despite the fact that Hub checkpoints may have `cache_implementation="hybrid"` (static sliding window).
        """
        model_id = "google/gemma-3-1b-it"
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

        # the default cache is static sliding window
        self.assertEqual(model.config.cache_implementation, "hybrid")
        self.assertEqual(model.generation_config.cache_implementation, "hybrid")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = "What is the capital of France?"
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        foward_outputs = model(**model_inputs)
        self.assertIn("DynamicSlidingWindowLayer", str(foward_outputs.past_key_values))

        generate_outputs = model.generate(
            **model_inputs, max_new_tokens=2, do_sample=False, return_dict_in_generate=True
        )
        self.assertIn("DynamicSlidingWindowLayer", str(generate_outputs.past_key_values))

        # If we manually specify the cache implementation = "hybrid", it will use the static sliding window cache
        generate_outputs = model.generate(
            **model_inputs,
            max_new_tokens=2,
            do_sample=False,
            return_dict_in_generate=True,
            cache_implementation="hybrid",
        )
        self.assertNotIn("DynamicSlidingWindowLayer", str(generate_outputs.past_key_values))
