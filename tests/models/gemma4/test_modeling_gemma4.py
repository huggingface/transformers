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

import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Gemma4Config,
    Gemma4TextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        Gemma4ForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Model,
        Gemma4Processor,
        Gemma4TextModel,
    )


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

    @unittest.skip(
        "Flaky on CI, but not locally on Mac. If model is set to fp32 instead of bf16, not flaky anymore."
        "TODO Cyril: investigate where the loss of precision between bf16 and fp32 comes from."
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Fails after fully removing the unused weights, even if `forward` is exactly the same. Investigate why."
    )
    def test_tp_generation_quantized(self):
        pass

    def test_model_training(self):
        pass

    @unittest.skip(
        "Under non-bf16 dtypes, MoE grouped_mm falls back to "
        "_grouped_mm_fallback_backward which is incompatible with torch.compile under 'reduce-overhead' mode"
    )
    def test_flash_attn_2_can_compile_with_attention_mask_None_without_graph_break(self):
        pass

    @unittest.skip(
        "Under non-bf16 dtypes, MoE grouped_mm falls back to "
        "_grouped_mm_fallback_backward which is incompatible with torch.compile under 'reduce-overhead' mode"
    )
    def test_torch_compile_for_training(self):
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

    def test_audio_rel_pos_encoding_uses_context_size_from_config(self):
        """Regression test for #45468; attention context size is properly read from config"""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioRelPositionalEncoding

        config = Gemma4AudioConfig(
            hidden_size=32,
            attention_chunk_size=6,
            attention_context_left=5,
            attention_context_right=1,
            use_clipped_linears=False,
        )

        module = Gemma4AudioRelPositionalEncoding(config)
        hidden_states = torch.zeros(1, 3, config.hidden_size)

        pos = module(hidden_states)

        context_size = config.attention_chunk_size + config.attention_context_left - 1 + config.attention_context_right
        expected_len = context_size // 2 + 1

        self.assertEqual(pos.shape, (1, expected_len, config.hidden_size))

        position_ids = torch.arange(context_size // 2, -1, -1, device=hidden_states.device)[..., None]
        scaled_time = position_ids * module.inv_timescales.to(device=hidden_states.device)
        expected = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1).to(hidden_states.dtype)

        torch.testing.assert_close(pos, expected)


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
        self.skip_flash_attn_inference_equivalence_tests()

    def skip_flash_attn_inference_equivalence_tests(self):
        skippable_tests = [
            "test_flash_attn_2_inference_equivalence",
            "test_flash_attn_3_inference_equivalence",
            "test_flash_attn_4_inference_equivalence",
        ]
        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(
                    reason="The base test does not pass image_position_ids and mm_token_type_ids required by Gemma4"
                )

    def test_training(self):
        # Overwrite to test training with text-only samples, should not raise errors
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        model = Gemma4ForConditionalGeneration(config)
        model.to(torch_device)
        model.train()
        inputs = self._prepare_for_class(inputs_dict, Gemma4ForConditionalGeneration, return_labels=True)
        loss = model(**inputs).loss
        loss.backward()

        # pop out image-related inputs and try to run forward
        inputs.pop("mm_token_type_ids", None)
        inputs.pop("pixel_values", None)
        loss = model(**inputs).loss
        loss.backward()

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

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        "Randomly starts failing after module order changed in the __init__ because accelertate is not robust enough"
    )
    def test_disk_offload_safetensors(self):
        pass


@slow
@require_torch_accelerator
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"
        self.processor = Gemma4Processor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_deterministic_for_xpu
    def test_model_with_image(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background'],
                ("xpu", 3): ['This image shows a **brown and white cow standing on a sandy beach near the ocean**.\n\nHere are some details about the image:\n\n*   '],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_with_image_batch(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": self.url1,
                    },
                    {"type": "image", "url": self.url2},
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

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are not identical.\n\nThe first image is a photograph of a **cow** standing on a beach under a blue sky.\n\n",
                ],
                ("cuda", (8, 6)): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are not identical.\n\nThe first image is a photograph of a **brown and white cow standing on a beach** under a blue",
                ],
                ("xpu", 3): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are not identical.\n\nThe first image is a photograph of a **brown and white cow standing on a beach** under a blue",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_multiimage(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url2},
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

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Traffic Sign:** The most prominent'],
                ("xpu", 3): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Roadway:** There is an'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_multi_gpu
    def test_model_text_only_multigpu(self):
        """Accelerate destroys the input dict `shared_kv_states` if it's not passed as kwarg and part of
        `_skip_keys_device_placement`, so test this to avoid regresions.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_deterministic_for_xpu
    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
                ("xpu", 3): ['## The Algorithmic Mind\n\nA whisper starts in silicon deep,\nWhere data streams in endless sweep.\nNo flesh and blood, no beating'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_states_sharing_with_and_without_cache(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Who are you? What can you do?"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)
        input_size = inputs.input_ids.shape[-1]

        # With and without cache generatiom should share kv states the same way
        output_with_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=True)
        output_without_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=False)

        output_text_with_cache = tokenizer.batch_decode(output_with_cache[:, input_size:], skip_special_tokens=True)
        output_text_without_cache = tokenizer.batch_decode(
            output_without_cache[:, input_size:], skip_special_tokens=True
        )

        self.assertEqual(output_text_with_cache, output_text_without_cache)

    # Note: we do not test FA2 as the head dim is 512 on some layers, which is not compatible with the kernels
    @parameterized.expand([("sdpa",), ("eager",)])
    @require_deterministic_for_xpu
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. Outputs for every attention functions
        should be coherent and identical.
        """

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding="left")
        input_text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in input_text
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = Gemma4ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=torch_device,
            attn_implementation=attn_implementation,
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, cache_implementation="static")
        output_text = tokenizer.batch_decode(out[:, input_size:])

        EXPECTED_COMPLETIONS = Expectations(
            {
                ("cuda", 8): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
                ("xpu", 3): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ],
            }
        )
        self.assertEqual(output_text, EXPECTED_COMPLETIONS.get_expectation())

    @pytest.mark.torch_export_test
    def test_export_text_only(self):
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        exportable_module = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=1024, device=torch_device
        )
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], device=torch_device, dtype=torch.long),
        )

        # Test generation with the exported model
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate, device=torch_device
        )

        input_text = tokenizer(prompt, return_tensors="pt").to(torch_device)
        eager_outputs = model.generate(
            **input_text,
            max_new_tokens=max_new_tokens_to_generate,
            do_sample=False,  # Use greedy decoding to match the exported model
        )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        self.assertEqual(export_generated_text, eager_generated_text)
