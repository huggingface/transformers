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
    # Xiaomi-Lens-0 uses the Qwen3-VL processing, the tiny Qwen3-VL processor avoids loading the full tokenizer
    tiny_model_id = "hf-internal-testing/tiny-processor-qwen3_vl"
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
        processor.tokenizer.add_tokens(["<|mask|>", "<think>", "</think>", "<bbox>", "</bbox>"], special_tokens=True)
        patch_size = processor.image_processor.patch_size

        texts = [
            # One mask while thinking, which is discarded, and two grounded instances in the answer
            "<think>\nThe cat <|mask|> is on the left.\n</think>\n\n"
            "<bbox>[0, 0, 500, 1000]</bbox><mask><|mask|></mask><bbox>[500, 250, 1000, 750]</bbox><mask><|mask|></mask>",
            # No thinking block, one grounded instance
            "<bbox>[100, 200, 300, 400]</bbox><mask><|mask|></mask>",
        ]
        input_ids = processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
        image_grid_thw = torch.tensor([[1, 2, 4], [1, 2, 2]])

        # Mask logits are positive on the left half of the first image, everywhere for the others
        pred_masks = torch.full((2, 3, 2 * patch_size, 4 * patch_size), 10.0)
        pred_masks[0, 1, :, 2 * patch_size :] = -10.0
        pred_masks[1, 0, :, 2 * patch_size :] = -10.0  # outside of the second image, ignored

        results = processor.post_process_grounding(
            input_ids, pred_masks, image_grid_thw, target_sizes=[(64, 128), (2 * patch_size, 2 * patch_size)]
        )

        torch.testing.assert_close(
            results[0]["boxes"], torch.tensor([[0.0, 0.0, 64.0, 64.0], [64.0, 16.0, 128.0, 48.0]])
        )
        self.assertEqual(results[0]["masks"].shape, (2, 64, 128))
        self.assertTrue(results[0]["masks"][0, :, :64].all())
        self.assertFalse(results[0]["masks"][0, :, 64:].any())
        self.assertTrue(results[0]["masks"][1].all())

        size = 2 * patch_size
        torch.testing.assert_close(results[1]["boxes"], torch.tensor([[0.1, 0.2, 0.3, 0.4]]) * size)
        self.assertEqual(results[1]["masks"].shape, (1, size, size))
        self.assertTrue(results[1]["masks"].all())
