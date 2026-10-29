# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import XiaomiLens0Processor


@require_vision
@require_torch
@require_torchvision
class XiaomiLens0ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = XiaomiLens0Processor
    tiny_model_id = "hf-internal-testing/tiny-processor-xiaomi_lens0"

    # Same settings/overrides as Qwen3. Only test_post_process_grounding is new.
    videos_unstructured_max_length = 870
    videos_text_kwargs_max_length = 870
    videos_text_kwargs_override_max_length = 870

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 96},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 72},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 48},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 48},
            {"expected_dim": 0, "output_length": 96},
        ]

    @unittest.skip("Xiaomi-Lens-0 can't sample frames from image frames directly")
    def test_apply_chat_template_video_1(self):
        pass

    @unittest.skip("Xiaomi-Lens-0 can't sample frames from image frames directly")
    def test_apply_chat_template_video_2(self):
        pass

    def test_post_process_grounding(self):
        processor = self.get_processor()
        height = 2 * processor.image_processor.patch_size
        width = 4 * processor.image_processor.patch_size
        image_grid_thw = torch.tensor([[1, 2, 4]])
        text = (
            "<think>The objects occupy <bbox>[0, 0, 1000, 1000]</bbox>.</think>\n\n"
            "The cat is at <bbox>[0, 0, 500, 1000]</bbox><mask><|mask|></mask>. "
            "The dog is at <bbox>[500, 250, 1000, 750]</bbox>.<|im_end|>"
        )

        expected_boxes = torch.tensor(
            [[0.0, 0.0, width / 2, float(height)], [width / 2, height / 4, float(width), 3 * height / 4]]
        )
        expected_mask = torch.zeros((1, height, width), dtype=torch.bool)
        expected_mask[:, :, : width // 2] = True
        mask_logits = torch.where(expected_mask, 10.0, -10.0)

        input_ids = processor.tokenizer(text, return_tensors="pt").input_ids
        results = processor.post_process_grounding(
            input_ids,
            image_grid_thw,
            mask_logits=[mask_logits],
            mask_token_indices=[(input_ids == processor.mask_token_id).nonzero()],
        )
        self.assertEqual(len(results), 1)
        torch.testing.assert_close(results[0]["boxes"], expected_boxes)
        torch.testing.assert_close(results[0]["masks"], expected_mask)

        # ignore_thinking=False must include the box in the thinking block.
        results = processor.post_process_grounding(
            input_ids,
            image_grid_thw,
            mask_logits=[mask_logits],
            mask_token_indices=[(input_ids == processor.mask_token_id).nonzero()],
            ignore_thinking=False,
        )
        thinking_box = torch.tensor([[0.0, 0.0, float(width), float(height)]])
        torch.testing.assert_close(results[0]["boxes"], torch.cat([thinking_box, expected_boxes]))
        torch.testing.assert_close(results[0]["masks"], expected_mask)
