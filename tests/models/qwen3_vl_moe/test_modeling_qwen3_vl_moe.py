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

from transformers import (
    AutoProcessor,
    Qwen3VLMoeConfig,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


class Qwen3VLMoeVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=16,
        text_config={
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 32,
            "vocab_size": 99,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "model_type": "qwen3_vl_moe",
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "moe_intermediate_size": 16,
            "num_experts_per_tok": 4,
            "num_experts": 8,
            "rope_theta": 10000,
            "tie_word_embeddings": True,
            "rope_parameters": {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=3,
        video_token_id=4,
        vision_start_token_id=5,
        vision_end_token_id=6,
        tie_word_embeddings=True,
        is_training=True,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.is_training = is_training

        self.vision_config = vision_config
        self.text_config = text_config

        self.vocab_size = text_config["vocab_size"]
        self.bos_token_id = text_config["bos_token_id"]
        self.eos_token_id = text_config["eos_token_id"]
        self.pad_token_id = text_config["pad_token_id"]
        self.hidden_size = text_config["hidden_size"]
        self.intermediate_size = text_config["intermediate_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_key_value_heads = text_config["num_key_value_heads"]
        self.rope_theta = text_config["rope_theta"]
        self.rope_parameters = text_config["rope_parameters"]
        self.hidden_act = text_config["hidden_act"]
        self.max_position_embeddings = text_config["max_position_embeddings"]
        self.model_type = text_config["model_type"]

        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Qwen3VLMoeConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
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

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Qwen3VLMoeModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Qwen3VLMoeForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Qwen3VLMoeModel,
            Qwen3VLMoeForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    def setUp(self):
        self.model_tester = Qwen3VLMoeVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3VLMoeConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
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
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_chans
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pathed_per_frame = patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )
        video_grid_thw = torch.tensor([[1, patch_H, patch_W] for _ in range(patch_T)] * B)

        # sanity check
        self.assertEqual(pixel_values_videos.shape[0], video_grid_thw.prod(dim=1).sum().item())

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_image_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_image_tokens

        self.assertLessEqual((B * patches_per_video) + insertion_point, self.model_tester.seq_length)
        for b in range(B):
            # each frame is separated by a vision_start_token_id
            for frame_idx in range(patch_T):
                input_ids[b, insertion_point + frame_idx * (pathed_per_frame + 1)] = (
                    self.model_tester.vision_start_token_id
                )
                input_ids[
                    b,
                    insertion_point + frame_idx * (pathed_per_frame + 1) + 1 : insertion_point
                    + (frame_idx + 1) * (pathed_per_frame + 1),
                ] = self.model_tester.video_token_id

        for model_class in self.all_model_classes:
            # TODO:we should remove this because we use timestamps for video
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            self.assertIsNotNone(outputs)


@require_torch
@unittest.skip("The checkpoint is not yet released")
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
        expected_input_ids = [151644, 872, 198, 151652, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655]  # fmt: skip
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
    @require_torch_gpu
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

        EXPECTED_DECODED_TEXT = [
            "user\nWhat kind of dog is this?\nassistant\nThis is a Pallas's cat, also known as the manul. It's a wild cat species native to the grasslands and montane regions",
            "user\nWhat kind of dog is this?\nassistant\nBased on the image provided, there is no dog present. The animals in the picture are two cats.\n\nHere are some observations about the cats in the"
        ]  # fmt: skip
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_flash_attn
    @require_torch_gpu
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
