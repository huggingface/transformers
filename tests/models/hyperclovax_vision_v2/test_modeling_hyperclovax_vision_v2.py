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

import tempfile
import unittest

import requests

from transformers import HCXVisionV2Config, is_torch_available, is_vision_available
from transformers.image_utils import load_image
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_cv2,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_cv2_available():
    import cv2

if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch

    from transformers import (
        HCXVisionV2Config,
        HCXVisionV2ForConditionalGeneration,
        HCXVisionV2ForSequenceClassification,
        HCXVisionV2Model,
        HCXVisionV2Processor,
        HyperCLOVAXConfig,
        HyperCLOVAXForCausalLM,
        HyperCLOVAXForSequenceClassification,
        HyperCLOVAXModel,
        Qwen2_5_VLVisionConfig,
    )


class HCXVisionV2VisionText2TextModelTester(VLMModelTester):
    base_model_class = HCXVisionV2Model
    config_class = HCXVisionV2Config
    text_config_class = HyperCLOVAXConfig
    vision_config_class = Qwen2_5_VLVisionConfig
    conditional_generation_class = HCXVisionV2ForConditionalGeneration
    sequence_classification_class = HCXVisionV2ForSequenceClassification

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 14)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_heads", 4)
        kwargs.setdefault("depth", 2)
        kwargs.setdefault("spatial_merge_size", 1)
        kwargs.setdefault("out_hidden_size", 32)
        kwargs.setdefault("tokens_per_second", 1)
        kwargs.setdefault("temporal_patch_size", 2)
        kwargs.setdefault("video_token_id", 4)

        super().__init__(parent, **kwargs)

    def get_vision_config(self):
        config = super().get_vision_config()
        return self.vision_config_class(
            **{
                **config.to_dict(),
                "patch_size": self.patch_size,
                "num_heads": self.num_heads,
                "depth": self.depth,
                "spatial_merge_size": self.spatial_merge_size,
                "out_hidden_size": self.out_hidden_size,
                "tokens_per_second": self.tokens_per_second,
                "temporal_patch_size": self.temporal_patch_size,
            }
        )

    def get_config(self):
        config = super().get_config()
        return self.config_class(
            **{
                **config.to_dict(),
                "video_token_id": self.video_token_id,
            }
        )

    def create_pixel_values(self):
        return floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (self.patch_size**2),
                self.num_channels * (self.patch_size**2) * self.temporal_patch_size,
            ]
        )

    def get_additional_inputs(self, config, input_ids, pixel_values):
        image_grid_thw = torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device)
        return {"image_grid_thw": image_grid_thw}


class HyperCLOVAXModelTester(CausalLMModelTester):
    base_model_class = HyperCLOVAXModel
    causal_lm_class = HyperCLOVAXForCausalLM
    sequence_classification_class = HyperCLOVAXForSequenceClassification


@require_torch
class HCXVisionV2ModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = HCXVisionV2VisionText2TextModelTester

    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.

        Override to handle image_grid_thw which is specific to HCXVisionV2.
        """
        import copy

        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # Test 1: remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            if "image_grid_thw" in curr_input_dict:
                curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            # First, take just the first item from each tensor
            curr_input_dict = {key: val[:1] for key, val in input_dict.items()}

            # Double the batch size for all batch-dimension tensors except pixel_values
            # This simulates having 2 prompts (each with image tokens) but only 1 image
            batch_tensors_to_double = ["input_ids", "attention_mask", "token_type_ids"]
            for key in batch_tensors_to_double:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: two images and two image tokens don't raise an error
            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]],
                dim=0,
            )
            if "image_grid_thw" in curr_input_dict:
                curr_input_dict["image_grid_thw"] = torch.cat(
                    [
                        curr_input_dict["image_grid_thw"],
                        curr_input_dict["image_grid_thw"],
                    ],
                    dim=0,
                )
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = torch.cat(
                    [curr_input_dict["image_sizes"], curr_input_dict["image_sizes"]],
                    dim=0,
                )
            _ = model(**curr_input_dict)

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_channels
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 4
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B * patches_per_video,
                # second dim: in_channels * temporal_patch_size * patch_size^2
                C * T * (P**2),
            ]
        )
        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * B)

        # sanity check
        assert pixel_values_videos.shape[0] == video_grid_thw.prod(dim=1).sum().item()

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_image_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_image_tokens

        assert (B * patches_per_video) + insertion_point <= self.model_tester.seq_length
        for b in range(B):
            input_ids[b, insertion_point : insertion_point + patches_per_video] = self.model_tester.video_token_id

        for model_class in self.all_model_classes:
            second_per_grid_ts = torch.tensor([1.0] * B, device=torch_device)
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            self.assertIsNotNone(outputs)

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        # Conversion happens only for the `ConditionalGeneration` model, not the base model
        try:
            self.all_model_classes = (
                (
                    HCXVisionV2ForConditionalGeneration,
                    HCXVisionV2ForSequenceClassification,
                )
                if is_torch_available()
                else ()
            )
            super().test_reverse_loading_mapping(check_keys_were_modified)
        finally:
            self.all_model_classes = (
                (
                    HCXVisionV2Model,
                    HCXVisionV2ForConditionalGeneration,
                    HCXVisionV2ForSequenceClassification,
                )
                if is_torch_available()
                else ()
            )

    @unittest.skip("Loading nested configs with overwritten `kwargs` isn't supported yet, FIXME @raushan.")
    def test_load_with_mismatched_shapes(self):
        pass

    @unittest.skip(reason="Inherits Qwen2_5_VL vision module with get_window_index incompatible with torch.export")
    def test_torch_export(self):
        pass


@require_torch
class HyperCLOVAXModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = HyperCLOVAXModelTester
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def test_reverse_loading_mapping(self, check_keys_were_modified=True):
        # Conversion happens only for the `ConditionalGeneration` model, not the base model
        try:
            self.all_model_classes = (
                (
                    HyperCLOVAXForCausalLM,
                    HyperCLOVAXForSequenceClassification,
                )
                if is_torch_available()
                else ()
            )
            super().test_reverse_loading_mapping(check_keys_were_modified)
        finally:
            self.all_model_classes = (
                (
                    HyperCLOVAXModel,
                    HyperCLOVAXForCausalLM,
                    HyperCLOVAXForSequenceClassification,
                )
                if is_torch_available()
                else ()
            )


@require_torch
@require_torch_accelerator
class HCXVisionV2IntegrationTest(unittest.TestCase):
    model_id = "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B"

    def setUp(self):
        self.processor = HCXVisionV2Processor.from_pretrained(self.model_id)
        self.model = HCXVisionV2ForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="auto"
        )
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_text_generate(self):
        messages = [
            {"role": "user", "content": "What is the capital of South Korea?"},
        ]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    "user\nWhat is the capital of South Korea?\nassistant\n<think>\n\n</think>\n\nThe capital of South Korea is **Seoul**."
                ],
            }
        )
        expected = EXPECTED_TEXTS.get_expectation()

        self.assertEqual(output_text, expected)

    @slow
    def test_image_generate(self):
        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        image = load_image(image_url).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What animal is in the image?"},
                ],
            }
        ]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    'user\n{"id": "image_00", "type": "image/jpeg", "filename": "image.jpg"}\n\nWhat animal is in the image?\nassistant\n<think>\n\n</think>\n\nThe animal in the image is a cow. It\'s a brown cow with a distinctive white stripe running down the middle of its face. The cow is'
                ],
            }
        )
        expected = EXPECTED_TEXTS.get_expectation()

        self.assertEqual(output_text, expected)

    @slow
    @require_cv2
    def test_video_generate(self):
        video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": video_url}},
                    {"type": "text", "text": "What is shown in this video?"},
                ],
            }
        ]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(requests.get(video_url).content)
            f.flush()
            cap = cv2.VideoCapture(f.name)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb).resize((224, 224), Image.BICUBIC))
            cap.release()

        inputs = self.processor(text=[text], videos=[frames], return_tensors="pt")
        inputs = inputs.to(torch_device)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    'user\n{"id": "video_00", "type": "video/mp4", "filename": "video.mp4"}\n<|video_aux_start|>다음 중 video_duration은 비디오 길이 정보입니다. 참고하여 답변하세요. {"video_duration": "<|video_meta_duration|>"}<|video_aux_end|>\n\n\nWhat is shown in this video?\nassistant\n<think>\n\n</think>\n\nThe video shows a person practicing tennis in an indoor court, focusing on their preparation and execution of serves.'
                ],
            }
        )
        expected = EXPECTED_TEXTS.get_expectation()

        self.assertEqual(output_text, expected)
