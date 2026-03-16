# Copyright 2026 NAVER Corp. and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch HyperClovaX Vision model."""

import copy
import unittest

from transformers import HCXVisionConfig, is_torch_available
from transformers.image_utils import load_image
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, torch_device
from transformers.video_utils import load_video

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import HCXVisionForConditionalGeneration, HCXVisionModel, HCXVisionV2Processor


class HCXVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        num_channels=3,
        ignore_index=-100,
        image_size=14,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        img_start_id=50,
        video_start_id=51,
        hidden_size=32,
        vocab_size=100,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        patch_size=14,
        temporal_patch_size=2,
        vision_depth=2,
        vision_hidden_size=32,
        vision_num_heads=4,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.ignore_index = ignore_index
        self.image_size = image_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.img_start_id = img_start_id
        self.video_start_id = video_start_id
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.vision_depth = vision_depth
        self.vision_hidden_size = vision_hidden_size
        self.vision_num_heads = vision_num_heads
        self.is_training = is_training

        # Number of image tokens for a single image_size x image_size image
        self.num_image_tokens = (image_size // patch_size) ** 2  # = 1 for image_size=14
        # seq_length is the TOTAL sequence length (text tokens + image placeholder tokens),
        # since the test framework uses this value to generate labels.
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return HCXVisionConfig(
            text_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
            },
            vision_config={
                "depth": self.vision_depth,
                "hidden_size": self.vision_hidden_size,
                "num_heads": self.vision_num_heads,
                "patch_size": self.patch_size,
                "spatial_merge_size": 1,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.num_channels,
                "intermediate_size": self.vision_hidden_size,
                "out_hidden_size": self.vision_hidden_size,
            },
            img_start_id=self.img_start_id,
            video_start_id=self.video_start_id,
            mm_projector_type="mlp",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # pixel_values: (num_patches, C * patch_size^2 * temporal_patch_size)
        num_patches = self.batch_size * self.num_image_tokens
        pixel_dim = self.num_channels * (self.patch_size**2) * self.temporal_patch_size
        pixel_values = floats_tensor([num_patches, pixel_dim])
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()

        # seq_length already includes image placeholder tokens
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        # Prevent collisions with special tokens
        input_ids[input_ids == self.img_start_id] = self.pad_token_id
        input_ids[input_ids == self.video_start_id] = self.pad_token_id
        # Insert img_start_id at position 0 for each sample in the batch
        input_ids[:, 0] = self.img_start_id

        inputs_dict = {
            "pixel_values": pixel_values.to(torch_device),
            "image_grid_thw": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "input_ids": input_ids.to(torch_device),
            "attention_mask": attention_mask.to(torch_device),
        }
        return config, inputs_dict


@require_torch
class HCXVisionForConditionalGenerationTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """Tests for HCXVisionForConditionalGeneration and HCXVisionModel."""

    all_model_classes = (HCXVisionModel, HCXVisionForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (HCXVisionForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = HCXVisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=HCXVisionConfig,
            has_text_modality=False,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mismatching_num_image_tokens(self):
        """
        VLMs must raise a clear error when the number of images does not match
        the number of image placeholder tokens in the input sequence.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            # Baseline forward should succeed
            _ = model(**input_dict)

            # Now mismatch: keep tokens but remove all but the last image
            bad_dict = copy.deepcopy(input_dict)
            bad_dict["pixel_values"] = bad_dict["pixel_values"][-1:]
            bad_dict["image_grid_thw"] = bad_dict["image_grid_thw"][-1:]
            with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
                _ = model(**bad_dict)

    def test_video_forward(self):
        """Forward pass with video inputs (pixel_values_videos, video_grid_thw)."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = self.model_tester.num_channels
        P = self.model_tester.patch_size
        T = self.model_tester.temporal_patch_size
        frames = 4

        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = max(1, frames // T)
        patches_per_video = patch_T * patch_H * patch_W
        pixel_dim = C * T * (P**2)

        pixel_values_videos = floats_tensor([B * patches_per_video, pixel_dim]).to(torch_device)
        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * B, device=torch_device)

        # Need patches_per_video video placeholder tokens per sequence.
        # The vision model produces patch_T * patch_H * patch_W tokens per video
        # (with spatial_merge_size=1, no reduction).
        total_seq_len = patches_per_video + self.model_tester.seq_length
        input_ids = ids_tensor([B, total_seq_len], self.model_tester.vocab_size).to(torch_device)
        # Avoid collision with video_start_id
        input_ids[input_ids == self.model_tester.video_start_id] = self.model_tester.pad_token_id
        # Insert patches_per_video video_start_id tokens at the front of each sequence
        for i in range(patches_per_video):
            input_ids[:, i] = self.model_tester.video_start_id

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
            self.assertIsNotNone(outputs)

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass


torch_device = "cpu"


@require_torch
@require_torch_accelerator
class HCXVisionIntegrationTest(unittest.TestCase):
    model_id = "/home/jp/DEMO/LLM42/base_models/HCX/HCX-SEED-Think-32B"

    def setUp(self):
        self.processor = HCXVisionV2Processor.from_pretrained(self.model_id)
        self.model = HCXVisionForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=torch_device,
        )

        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
        self.image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "data": {"url": image_url}},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]
        self.video_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": video_url}},
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]
        self.image = load_image(url_to_local_path(image_url)).convert("RGB")
        self.video, _ = load_video(url_to_local_path(video_url))
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_image_logits(self):
        text = self.processor.apply_chat_template(self.image_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        with torch.no_grad():
            output = self.model(**inputs)

        EXPECTED_LOGITS_SLICE = torch.tensor(
            [
                [2.1406, 4.7500, 5.1562],
                [-1.5391, 5.4688, 5.5312],
                [-1.3672, -4.4688, -1.8281],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(output.logits[0, :3, :3].float().cpu(), EXPECTED_LOGITS_SLICE, atol=1e-2, rtol=1e-3)

    def test_model_image_generate(self):
        text = self.processor.apply_chat_template(self.image_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt").to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        EXPECTED_DECODED_TEXT = "This image shows a cow standing on a sandy beach."

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        self.assertTrue(EXPECTED_DECODED_TEXT in decoded)

    def test_model_video_generate(self):
        text = self.processor.apply_chat_template(self.video_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[self.video], return_tensors="pt").to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        EXPECTED_DECODED_TEXT = "This image shows a cow standing on a sandy beach."

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        self.assertTrue(EXPECTED_DECODED_TEXT in decoded)
