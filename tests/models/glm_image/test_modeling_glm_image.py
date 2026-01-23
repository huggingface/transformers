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
"""Testing suite for the PyTorch GLM-Image model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    GlmImageConfig,
    GlmImageForConditionalGeneration,
    GlmImageModel,
    is_torch_available,
)
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    run_first,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


class GlmImageVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=128,
        image_start_token_id=85,
        image_end_token_id=86,
        image_token_id=7,
        is_training=True,
        text_config={
            "vocab_size": 99,
            "vision_vocab_size": 99,
            "hidden_size": 16,
            "intermediate_size": 22,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "output_channels": 64,
            "hidden_act": "silu",
            "max_position_embeddings": 512,
            "rope_parameters": {"type": "default", "mrope_section": [2, 1, 1]},
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "bos_token_id": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "n_group": 1,
            "topk_group": 1,
            "num_experts_per_tok": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_act": "gelu",
            "hidden_size": 32,
            "out_hidden_size": 16,
            "intermediate_size": 22,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
        },
        vq_config={
            "embed_dim": 48,
            "in_channels": 3,
            "initializer_range": 0.02,
            "latent_channels": 32,
            "num_embeddings": 32,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.image_token_id = image_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.vq_config = vq_config
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.vision_vocab_size = text_config["vision_vocab_size"]
        self.vocab_size = text_config["vocab_size"]
        self.num_image_tokens = 64
        self.seq_length = seq_length + self.num_image_tokens
        self.n_routed_experts = text_config["n_routed_experts"]
        self.n_shared_experts = text_config["n_shared_experts"]
        self.num_experts_per_tok = text_config["num_experts_per_tok"]
        self.n_group = text_config["n_group"]
        self.topk_group = text_config["topk_group"]

    def get_config(self):
        return GlmImageConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vq_config=self.vq_config,
            image_token_id=self.image_token_id,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_end_token_id] = self.pad_token_id

        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        patch_size = config.vision_config.patch_size
        patches_per_side = self.image_size // patch_size

        # For i2i mode: each sample has 1 source image + 1 target grid
        # image_grid_thw layout: [sample0_source, sample0_target, sample1_source, sample1_target, ...]
        # Since batches are homogeneous, all samples have same number of source images
        num_grids_per_sample = 2  # 1 source + 1 target
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * (self.batch_size * num_grids_per_sample),
                device=torch_device,
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images_per_sample": torch.tensor([num_grids_per_sample] * self.batch_size, device=torch_device),
        }
        return config, inputs_dict


@require_torch
class GlmImageModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (GlmImageModel, GlmImageForConditionalGeneration) if is_torch_available() else ()

    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5
    _is_composite = True

    def setUp(self):
        self.model_tester = GlmImageVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GlmImageConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    # GlmImage has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]

        # The diff from the general `prepare_config_and_inputs_for_generate` lies here
        patch_size = config.vision_config.patch_size
        num_patches_per_image = (self.model_tester.image_size**2) // (patch_size**2)
        num_grids_per_sample = 2  # 1 source + 1 target

        filtered_inputs_dict = {
            k: v[:batch_size, ...]
            if isinstance(v, torch.Tensor) and k not in ["pixel_values", "image_grid_thw", "images_per_sample"]
            else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        # pixel_values: each sample has 1 source image
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][: batch_size * num_patches_per_image]
        # image_grid_thw: each sample has 2 grids (1 source + 1 target)
        filtered_inputs_dict["image_grid_thw"] = inputs_dict["image_grid_thw"][: batch_size * num_grids_per_sample]
        # images_per_sample: each sample has 2 images
        filtered_inputs_dict["images_per_sample"] = torch.tensor(
            [num_grids_per_sample] * batch_size, device=torch_device
        )

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def test_training(self):
        # Model isn't in any auto-mapping so we need to build labels manually
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
            )
            loss = model(**inputs_dict).loss
            loss.backward()

    @unittest.skip(reason="Reequires input ids AND image grid to generate")
    def test_generate_without_input_ids(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("Needs special input preparation. Not important test for model, skip for now")
    def test_eager_matches_sdpa_inference(
        self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

    @unittest.skip(reason="No available kernels - not supported")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Size mismatch")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @pytest.mark.xfail(
        reason="GlmImage has a VQ module that uses `weight.data` directly in forward which prevent offloading on that module"
    )
    def test_disk_offload_safetensors(self):
        pass

    @pytest.mark.xfail(
        reason="GlmImage has a VQ module that uses `weight.data` directly in forward which prevent offloading on that module"
    )
    def test_disk_offload_bin(self):
        pass

    @pytest.mark.xfail(
        reason="GlmImage has a VQ module that uses `weight.data` directly in forward which prevent offloading on that module"
    )
    def test_cpu_offload(self):
        pass

    @pytest.mark.xfail(
        reason="GlmImage has a VQ module that uses `weight.data` directly in forward which prevent offloading on that module"
    )
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="GLM-Image has special image token IDs that get clamped when vocab is resized smaller")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(reason="GLM-Image has special image token IDs that get clamped when vocab is resized smaller")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip("Error with compilation")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(reason="GLM-Image does not use inputs_embeds")
    def test_generate_from_inputs_embeds(self, _, num_beams):
        pass

    @unittest.skip(reason="GLM-Image input embed is compare with inputs_ids and image_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="GLM-Image does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="GLM-Image can't do text-only inference")
    def test_generate_from_random_inputs_embeds(self):
        pass

    @unittest.skip(reason="GLM-Image can't do and does not need assisted generation. Not worth fixing!")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="GLM-Image can't do and does not need assisted generation. Not worth fixing!")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip(reason="GLM-Image can't do and does not need assisted generation. Not worth fixing!")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip(reason="GlmImageVisionModel does not support training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="GlmImageVision does not support output_hidden_states test")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="GlmImageVisionModel does not support training")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="GlmImage needs special input preparation to pass this test")
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    @unittest.skip(
        reason="GlmImage is a multimodal model that requires pixel_values and image_grid_thw. "
        "This test drops all inputs except input_ids which causes NoneType iteration error."
    )
    def test_flash_attention_2_continue_generate_with_position_ids(self):
        pass

    @unittest.skip(
        reason="GlmImage is a multimodal model that requires pixel_values and image_grid_thw. "
        "This test only uses input_ids and attention_mask which causes NoneType iteration error."
    )
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        reason="GlmImage is a multimodal model that requires pixel_values and image_grid_thw. "
        "This test only uses input_ids and attention_mask which causes NoneType iteration error."
    )
    def test_flash_attn_2_from_config(self):
        pass


@require_torch
@slow
class GlmImageIntegrationTest(unittest.TestCase):
    model_id = "zai-org/GLM-Image/vision_language_encoder"
    processor_id = "zai-org/GLM-Image/processor"

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = GlmImageForConditionalGeneration.from_pretrained(
                cls.model_id, torch_dtype=torch.bfloat16, device_map="auto"
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "model"):
            del cls.model
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.processor = AutoProcessor.from_pretrained(self.processor_id)
        # Text-to-image generation message
        self.t2i_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A cute cat sitting on a wooden table"},
                ],
            }
        ]
        # Image-to-image generation message
        self.i2i_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "Add a red hat to this cat"},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_processor_text_to_image(self):
        """Test processor correctly prepares text-to-image inputs."""
        inputs = self.processor.apply_chat_template(
            self.t2i_message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        # For T2I, there should be no pixel_values (no source images)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("image_grid_thw", inputs)
        # T2I should have 2 target grids (main + prev for coarse-to-fine generation)
        self.assertEqual(inputs["image_grid_thw"].shape[0], 2)

    def test_processor_image_to_image(self):
        """Test processor correctly prepares image-to-image inputs."""
        inputs = self.processor.apply_chat_template(
            self.i2i_message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        # For I2I, there should be pixel_values from the source image
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_grid_thw", inputs)
        # I2I should have 1 source grid + 1 target grid = 2 grids
        self.assertEqual(inputs["image_grid_thw"].shape[0], 2)
        # images_per_sample should be 2 (1 source + 1 target)
        self.assertEqual(inputs["images_per_sample"].item(), 2)

    def test_text_to_image_generation(self):
        """Test text-to-image generation produces valid image tokens."""
        model = self.get_model()
        inputs = self.processor.apply_chat_template(
            self.t2i_message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # Generate image tokens
        output = model.generate(**inputs, max_new_tokens=10)

        # Output should be longer than input (generated tokens)
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])
        # Generated tokens should be within vision vocabulary range
        generated_tokens = output[0, inputs["input_ids"].shape[1] :]
        # Vision tokens are in range [image_start_token_id, image_end_token_id)
        self.assertTrue(all(t.item() < model.config.text_config.vision_vocab_size for t in generated_tokens))

    def test_image_to_image_generation(self):
        """Test image-to-image generation produces valid image tokens."""
        model = self.get_model()
        inputs = self.processor.apply_chat_template(
            self.i2i_message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # Generate image tokens
        output = model.generate(**inputs, max_new_tokens=10)

        # Output should be longer than input (generated tokens)
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])

    @run_first
    @require_flash_attn
    @require_torch_accelerator
    def test_flash_attention_generation(self):
        """Test generation with Flash Attention 2."""
        model = GlmImageForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        inputs = self.processor.apply_chat_template(
            self.t2i_message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # Generate image tokens
        output = model.generate(**inputs, max_new_tokens=5)

        # Output should be longer than input
        self.assertGreater(output.shape[1], inputs["input_ids"].shape[1])
