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
        batch_size=1,
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

        # Key fix: image_grid_thw should have batch_size rows for input images
        # plus 1 extra row that will be skipped by model's [:-1] slicing
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size
                + [[1, patches_per_side, patches_per_side]],  # Extra row for model's [:-1]
                device=torch_device,
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
        filtered_image_length = batch_size * (self.model_tester.image_size**2) // (patch_size**2)
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, torch.Tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }
        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:filtered_image_length]
        filtered_inputs_dict["image_grid_thw"] = inputs_dict["image_grid_thw"][: batch_size + 1]

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
    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = GlmImageForConditionalGeneration.from_pretrained(
                "zai-org/GLM-4.5V", dtype="auto", device_map="auto"
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "model"):
            del cls.model
        cleanup(torch_device, gc_collect=True)

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.processor = AutoProcessor.from_pretrained(
            "zai-org/GLM-4.5V", size={"shortest_edge": 10800, "longest_edge": 10800}
        )
        self.message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        self.message2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        self.message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [151331, 151333, 151336, 198, 151339, 151363, 151363, 151363, 151363, 151363, 151363,
                              151340, 3838, 3093, 315, 5562, 374]  # fmt: skip
        assert expected_input_ids == inputs.input_ids[0].tolist()[:17]

        expected_pixel_slice = torch.tensor(
            [
                [-0.1134, -0.4492, -0.8580],
                [-0.6244, -1.1645, -0.7120],
                [-0.3324, -0.7996, -0.7120],
                [0.2077, 0.2223, 0.4121],
                [0.4413, 0.1931, 0.4559],
                [0.5873, 0.3099, 0.4851],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        torch.testing.assert_close(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=1e-4, rtol=1e-4)

    def test_small_model_integration_test_batch(self):
        model = self.get_model()
        batch_messages = [self.message, self.message2, self.message_wo_image]
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=10)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it, let's try to figure out",
            "\nWhat kind of dog is this?\n<think>Got it, let's see. The user",
            '\nWho are you?\n<think>The user is asking "Who are you?"'
        ]  # fmt: skip
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)
        decoded = [x.replace("<|image|>", "") for x in decoded]
        self.assertEqual(
            decoded,
            EXPECTED_DECODED_TEXT,
        )

    @run_first
    @require_flash_attn
    @require_torch_accelerator
    def test_small_model_integration_test_batch_flashatt2(self):
        model = GlmImageForConditionalGeneration.from_pretrained(
            "zai-org/GLM-4.5V",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        batch_messages = [self.message, self.message2, self.message_wo_image]
        inputs = self.processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=3)

        EXPECTED_DECODED_TEXT = [
            "\nWhat kind of dog is this?\n<think>Got it",
            "\nWhat kind of dog is this?\n<think>Got it",
            "\nWho are you?\n<think>The user",
        ]  # fmt: skip
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)
        decoded = [x.replace("<|image|>", "") for x in decoded]
        self.assertEqual(
            decoded,
            EXPECTED_DECODED_TEXT,
        )
