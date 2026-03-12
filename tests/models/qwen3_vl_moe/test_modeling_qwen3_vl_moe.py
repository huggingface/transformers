# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3VLMoe model."""

import copy
import unittest

import pytest

from transformers import (
    AutoProcessor,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    is_torch_available,
)
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig, Qwen3VLMoeVisionConfig
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class Qwen3VLMoeVisionText2TextModelTester(VLMModelTester):
    base_model_class = Qwen3VLMoeModel
    config_class = Qwen3VLMoeConfig
    text_config_class = Qwen3VLMoeTextConfig
    vision_config_class = Qwen3VLMoeVisionConfig
    conditional_generation_class = Qwen3VLMoeForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("vision_start_token_id", 5)
        kwargs.setdefault("vision_end_token_id", 6)
        kwargs.setdefault("image_size", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("num_experts_per_tok", 4)
        kwargs.setdefault("num_experts", 8)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("vision_hidden_act", "gelu_pytorch_tanh")
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("num_position_embeddings", 16)
        kwargs.setdefault("deepstack_visual_indexes", [0, 1])
        kwargs.setdefault(
            "rope_parameters", {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True}
        )
        kwargs.setdefault("decoder_sparse_step", 1)
        super().__init__(parent, **kwargs)

        # Qwen3 VL MoE-specific vision config attributes (computed from instance state)
        self.in_channels = self.num_channels
        self.out_hidden_size = self.hidden_size
        self.vision_hidden_size = self.hidden_size
        self.vision_intermediate_size = self.hidden_size

    def create_pixel_values(self):
        """Qwen3 VL MoE expects flattened patches: (total_patches, channels * patch_size^2 * temporal_patch_size)"""
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        # Clear any accidental special tokens first
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        # Place image tokens with vision_start_token_id prefix
        input_ids[:, -1] = self.image_token_id
        input_ids[:, -2] = self.vision_start_token_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        # Qwen3VL requires image_grid_thw tensor
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_token_id] = 1
        return {
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_config(self):
        # Qwen3VLMoeConfig expects text_config and vision_config as dicts, not config objects
        return self.config_class(
            text_config=self.get_text_config().to_dict(),
            vision_config=self.get_vision_config().to_dict(),
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
        )


@require_torch
class Qwen3VLMoeModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Qwen3VLMoeVisionText2TextModelTester

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing(self):
        super().test_training_gradient_checkpointing()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        super().test_training_gradient_checkpointing_use_reentrant_false()

    @pytest.mark.xfail(reason="This architecture seems to not compute gradients for some layer.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        super().test_training_gradient_checkpointing_use_reentrant_true()

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

    def test_image_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size
        num_images = 2

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        # For this tiny config, each image corresponds to one patch token.
        patches_per_image = 1
        pixel_values = floats_tensor(
            [
                B * num_images * patches_per_image,
                C * T * (P**2),
            ]
        )
        image_grid_thw = torch.tensor([[1, 1, 1]] * (B * num_images))
        self.assertEqual(pixel_values.shape[0], image_grid_thw.prod(dim=1).sum().item())

        insertion_point = 0
        tokens_per_image = 3  # vision_start + image_token + vision_end
        required_seq_length = insertion_point + num_images * tokens_per_image
        self.assertLessEqual(required_seq_length, input_ids.shape[1])

        for b in range(B):
            for image_idx in range(num_images):
                image_start = insertion_point + image_idx * tokens_per_image
                input_ids[b, image_start] = self.model_tester.vision_start_token_id
                input_ids[b, image_start + 1] = self.model_tester.image_token_id
                input_ids[b, image_start + 2] = self.model_tester.vision_end_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.image_token_id] = 1

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        num_video = 2
        frame_timestamp_tokens = 5
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pathed_per_frame = patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * num_video * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )

        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * (B * num_video))

        # sanity check
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id

        insertion_point = 0
        tokens_per_frame = frame_timestamp_tokens + 1 + pathed_per_frame + 1
        tokens_per_video = patch_T * tokens_per_frame
        required_seq_length = insertion_point + num_video * tokens_per_video
        if required_seq_length > input_ids.shape[1]:
            pad_extension = torch.full(
                (B, required_seq_length - input_ids.shape[1]),
                self.model_tester.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad_extension], dim=1)
        timestamp_start_token_id = self.model_tester.vision_end_token_id + 1
        self.assertLessEqual(timestamp_start_token_id + frame_timestamp_tokens, self.model_tester.vocab_size)
        timestamp_token_ids = torch.arange(
            timestamp_start_token_id,
            timestamp_start_token_id + frame_timestamp_tokens,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        self.assertLessEqual(required_seq_length, input_ids.shape[1])
        for b in range(B):
            for video_idx in range(num_video):
                video_start = insertion_point + video_idx * tokens_per_video
                for frame_idx in range(patch_T):
                    frame_start = video_start + frame_idx * tokens_per_frame
                    input_ids[b, frame_start : frame_start + frame_timestamp_tokens] = timestamp_token_ids

                    vision_start_pos = frame_start + frame_timestamp_tokens
                    input_ids[b, vision_start_pos] = self.model_tester.vision_start_token_id

                    frame_token_start = vision_start_pos + 1
                    frame_token_end = frame_token_start + pathed_per_frame
                    input_ids[b, frame_token_start:frame_token_end] = self.model_tester.video_token_id

                    input_ids[b, frame_token_end] = self.model_tester.vision_end_token_id

        # build mm_token_type_ids
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.model_tester.video_token_id] = 2

        for model_class in self.all_model_classes:
            # TODO:we should remove this because we use timestamps for video
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.assertIsNotNone(outputs)

    # Need to be False as we only use a Transpose without modifying the keys
    def test_reverse_loading_mapping(self, check_keys_were_modified=False):
        super().test_reverse_loading_mapping(check_keys_were_modified)


@require_torch
class Qwen3VLMoeIntegrationTest(unittest.TestCase):
    def setUp(self):
        cleanup(torch_device, gc_collect=True)

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        self.processor.tokenizer.padding_side = "left"
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
        self.message3 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
                    },
                    {"type": "text", "text": "Describe the video in short."},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )

        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        expected_input_ids = [151644, 872, 198, 151652, 151655, 151655, 151655, 151655,
                              151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655]  # fmt: skip
        self.assertListEqual(expected_input_ids, inputs.input_ids[0].tolist()[:17])

        expected_pixel_slice = torch.tensor(
            [
                [-0.0902, -0.0824, -0.0824],
                [-0.2627, -0.2627, -0.2627],
                [-0.0824, -0.0902, -0.0902],
                [-0.0118, -0.0510, -0.1137],
                [-0.5137, -0.5529, -0.6078],
                [-0.6941, -0.6314, -0.5765],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        self.assertTrue(torch.allclose(expected_pixel_slice, inputs.pixel_values[:6, :3], atol=3e-3))

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        EXPECTED_DECODED_TEXT = "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a small wild cat native to the grasslands and steppes"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        batch_messages = [self.message] * 2
        inputs = self.processor.apply_chat_template(
            batch_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a small wild cat native to the grasslands and montane regions",
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a small wild cat native to the grasslands and montane regions"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_with_video(self):
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", max_image_size={"longest_edge": 50176}
        )
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype=torch.float16, device_map="auto"
        )
        questions = ["How long is the video? Describe the it in short."]
        video_urls = ["https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"]
        messages = [
            [
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
            for question, video_url in zip(questions, video_urls)
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        EXPECTED_DECODED_TEXT = ["user\n<0.3 seconds><1.4 seconds><2.5 seconds><3.6 seconds><4.7 seconds><5.8 seconds>How long is the video? Describe the it in short.\nassistant\nThe video is 6 seconds long. It shows a man playing tennis on an indoor court. He is wearing a white shirt and black shorts. He"]  # fmt: skip

        self.assertEqual(
            processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        inputs = self.processor.apply_chat_template(
            self.message, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThe animal in the image is not a dog. It is a **Pallas's cat** (*Otocolobus manul*), also known",
            "user\nWhat kind of dog is this?\nassistant\nThe animal in the image is not a dog. It is a **Pallas's cat** (also known as the manul), a wild f"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_expand_with_video(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        inputs = self.processor.apply_chat_template(
            self.message3, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False, num_beams=2, num_return_sequences=2)

        EXPECTED_DECODED_TEXT = [
            "user\n<0.3 seconds><1.3 seconds><2.4 seconds><3.5 seconds><4.6 seconds><5.6 seconds><6.7 seconds><7.8 seconds><8.9 seconds><9.7 seconds>Describe the video in short.\nassistant\nA baby wearing glasses sits on a bed and flips through a book.",
            "user\n<0.3 seconds><1.3 seconds><2.4 seconds><3.5 seconds><4.6 seconds><5.6 seconds><6.7 seconds><7.8 seconds><8.9 seconds><9.7 seconds>Describe the video in short.\nassistant\nA baby wearing glasses sits on a bed and flips through the pages of a book."
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_wo_image(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]
        batched_messages = [self.message, message_wo_image]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a wild cat species native to the grasslands and steppes",
            "user\nWho are you?\nassistant\nI am Qwen, a large-scale language model developed by Alibaba Cloud's Tongyi Lab. I can assist you with answering questions, creating text such"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch_different_resolutions(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
        )
        batched_messages = [self.message, self.message2]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a wild cat species native to the grasslands and steppes",
            "user\nWhat kind of dog is this?\nassistant\nBased on the image provided, the animals are not dogs. They are two cats.\n\nHere is a description of the animals in the image:\n\n-  "
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_flashatt2(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        batched_messages = [self.message, self.message2]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        # fmt: off
        EXPECTED_DECODED_TEXTS = Expectations(
            {
                (None, None): ["user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a wild cat species native to the grasslands and montane regions",
                               "user\nWhat kind of dog is this?\nassistant\nBased on the image provided, there is no dog present. The animals in the picture are two cats.\n\nHere are some observations about the cats in the"
                              ],
                ("xpu", None): ["user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a small wild cat native to the grasslands and steppes",
                                'user\nWhat kind of dog is this?\nassistant\nBased on the image provided, there is no dog present. The animals in the picture are two cats.\n\nHere is a description of the scene:\n-'
                              ],
            }
        )
        EXPECTED_DECODED_TEXT = EXPECTED_DECODED_TEXTS.get_expectation()
        # fmt: on

        DECODED_TEXT = self.processor.batch_decode(output, skip_special_tokens=True)
        self.assertEqual(
            DECODED_TEXT,
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    def test_small_model_integration_test_batch_wo_image_flashatt2(self):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        message_wo_image = [
            {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
        ]
        batched_messages = [self.message, message_wo_image]
        inputs = self.processor.apply_chat_template(
            batched_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(torch_device)

        # it should not matter whether two images are the same size or not
        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a wild cat species native to the grasslands and montane regions",
            "user\nWho are you?\nassistant\nI am Qwen, a large-scale language model developed by Alibaba Cloud's Tongyi Lab. I can assist you with answering questions, creating text such"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
