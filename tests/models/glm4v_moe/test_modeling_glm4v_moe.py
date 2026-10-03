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
"""Testing suite for the PyTorch GLM-4.5V model."""

import copy
import tempfile
import unittest

from transformers import (
    AutoProcessor,
    Glm4vMoeConfig,
    Glm4vMoeForConditionalGeneration,
    Glm4vMoeModel,
    is_torch_available,
)
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeTextConfig, Glm4vMoeVisionConfig
from transformers.testing_utils import (
    backend_device_count,
    get_cpu_ram_total_gib,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    run_first,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Glm4vMoeVisionText2TextModelTester(VLMModelTester):
    base_model_class = Glm4vMoeModel
    config_class = Glm4vMoeConfig
    text_config_class = Glm4vMoeTextConfig
    vision_config_class = Glm4vMoeVisionConfig
    conditional_generation_class = Glm4vMoeForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("video_start_token_id", 3)
        kwargs.setdefault("video_end_token_id", 4)
        kwargs.setdefault("image_start_token_id", 5)
        kwargs.setdefault("image_end_token_id", 6)
        kwargs.setdefault("image_token_id", 7)
        kwargs.setdefault("video_token_id", 8)
        kwargs.setdefault("image_size", 112)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_image_tokens", 64)
        kwargs.setdefault("seq_length", 128)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 0)
        kwargs.setdefault("intermediate_size", 22)
        kwargs.setdefault("num_key_value_heads", 1)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.5, "rope_theta": 10000},
        )
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("n_routed_experts", 8)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("n_group", 1)
        kwargs.setdefault("topk_group", 1)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.video_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        }

    def get_vision_config(self):
        return self.vision_config_class(
            depth=self.depth,
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            out_hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            temporal_patch_size=self.temporal_patch_size,
        )

    def create_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, 0] = self.image_start_token_id
        input_ids[:, 1 : 1 + self.num_image_tokens] = self.image_token_id
        input_ids[:, 1 + self.num_image_tokens] = self.image_end_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        patches_per_side = self.image_size // self.patch_size
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor(
                [[1, patches_per_side, patches_per_side]] * self.batch_size, device=torch_device
            ),
            "mm_token_type_ids": mm_token_type_ids,
        }


@require_torch
class Glm4vMoeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Glm4vMoeVisionText2TextModelTester
    model_split_percents = [0.7, 0.9]  # model too big to split at 0.5

    @unittest.skip("We don't really care about this one, test is not that slow")
    def test_model_is_small(self):
        pass

    # Glm4vMoe has images shaped as (bs*patch_len, dim) so we can't slice to batches in generate
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

    @unittest.skip(reason="No available kernels - not supported")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def test_mismatching_num_image_tokens(self):
        # Override the base test because we need to slice image_grid_thw too
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**curr_input_dict)

            model.base_model.rope_deltas = None
            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            mm_token_type_ids = curr_input_dict["mm_token_type_ids"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=torch.cat([mm_token_type_ids, mm_token_type_ids], dim=0),
                )

            model.base_model.rope_deltas = None
            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            mm_token_type_ids = torch.cat(
                [curr_input_dict["mm_token_type_ids"][:1], curr_input_dict["mm_token_type_ids"][:1]], dim=0
            )
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)
            with torch.no_grad():
                model(**inputs)[0]

    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_grid_thw"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)


@require_torch
@slow
class Glm4vMoeIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = None
        cls.offload_dir = None

    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.offload_dir = tempfile.TemporaryDirectory()
            # device_map="auto" fills GPUs to ~100%, leaving no room for the ~1.4 GiB
            # MergeModulelist temporary buffer that fuses per-expert weight shards into
            # a single gate_up_proj tensor during from_pretrained — causing CUDA OOM on
            # multi-GPU runners. A 70% per-GPU max_memory cap reserves the headroom.
            n = backend_device_count(torch_device)
            if n > 0 and torch_device != "cpu":
                torch_accel = getattr(torch, torch_device)
                per_device = int(
                    min(torch_accel.get_device_properties(i).total_memory for i in range(n)) * 0.70 / 1024**3
                )
                max_memory = dict.fromkeys(range(n), f"{per_device}GiB")
                max_memory["cpu"] = (
                    f"{int(get_cpu_ram_total_gib() * 0.9)}GiB"  # To avoid runner failing with exit code 137.
                )
            else:
                max_memory = None
            cls.model = Glm4vMoeForConditionalGeneration.from_pretrained(
                "zai-org/GLM-4.5V",
                dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                offload_folder=cls.offload_dir.name,
            )
        return cls.model

    @classmethod
    def tearDownClass(cls):
        if cls.offload_dir is not None:
            cls.offload_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(
            "zai-org/GLM-4.5V", size={"shortest_edge": 10800, "longest_edge": 10800}
        )
        self.message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/pipeline-cat-chonk.jpeg"
                        ),
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
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/coco_sample.png"
                        ),
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]
        self.message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]

        question = "Describe this video."
        video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
        self.video_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_url,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

    def test_small_model_integration_test(self):
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [151331, 151333, 151336, 198, 151339, 151363, 151363, 151363, 151363, 151363, 151363, 151340, 3838, 3093, 315, 5562, 374]  # fmt: skip
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
            "\nWhat kind of dog is this?\n<think>Got it, let's see. The question",
            '\nWho are you?\n<think>The user is asking "Who are you?"'
        ]  # fmt: skip
        decoded = self.processor.batch_decode(output, skip_special_tokens=True)
        decoded = [x.replace("<|image|>", "") for x in decoded]
        self.assertEqual(
            decoded,
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained("zai-org/GLM-4.5V", max_image_size={"longest_edge": 50176})
        model = self.get_model()
        batch_messages = [self.video_messages]
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=3)
        EXPECTED_DECODED_TEXT = ["\n012345Describe this video.\n<think>Got it"]  # fmt: skip
        decoded = processor.batch_decode(output, skip_special_tokens=True)
        decoded = [x.replace("<|image|>", "") for x in decoded]
        self.assertEqual(
            decoded,
            EXPECTED_DECODED_TEXT,
        )

    @run_first
    @require_flash_attn
    @require_torch_accelerator
    def test_small_model_integration_test_batch_flashatt2(self):
        with tempfile.TemporaryDirectory() as offload_dir:
            model = Glm4vMoeForConditionalGeneration.from_pretrained(
                "zai-org/GLM-4.5V",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                offload_folder=offload_dir,
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
