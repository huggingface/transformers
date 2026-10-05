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

import unittest

from parameterized import parameterized

from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Kimi_K25Processor


@require_vision
@require_torch
@require_torchvision
class Kimi_K25ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Kimi_K25Processor
    # Tiny processor created with make_tiny_processor.py from "RaushanTurganbay/kimi2.7-processor"
    tiny_model_id = "hf-internal-testing/tiny-processor-kimi_k25"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(model_id, trust_remote_code=False, **kwargs)

    @classmethod
    def _setup_video_processor(cls):
        # Small spatial size (28×28) and patch sizes keep video tensor allocations minimal.
        video_processor_class = cls._get_component_class_from_processor("video_processor")
        video_processor_kwargs = {
            "size": {"max_height": 28, "max_width": 28},
            "patch_size": 4,
            "temporal_patch_size": 2,
        }
        return video_processor_class(**video_processor_kwargs)

    @classmethod
    def _setup_image_processor(cls):
        # Small spatial size (28×28) and patch size keep image tensor allocations minimal.
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        image_processor_kwargs = {
            "size": {"max_height": 28, "max_width": 28},
            "patch_size": 4,
        }
        return image_processor_class(**image_processor_kwargs)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.video_token = processor.video_token

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 1848},
            {"num_frames": None, "fps": 16, "expected_dim": 0, "output_length": 3080},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 6776},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 6776},
        ]

    def test_kwargs_overrides_custom_image_processor_kwargs(self):
        processor = self.get_processor()

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_images_inputs()
        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(inputs[self.images_input_name].shape[0], 56)
        inputs = processor(
            text=input_str,
            images=image_input,
            size={"max_height": 56 * 56 * 4, "max_width": 56 * 56 * 4},
            return_tensors="pt",
        )
        self.assertEqual(inputs[self.images_input_name].shape[0], 800)

    @parameterized.expand([(1, "pt")])
    @unittest.skip("Kimi sampels with FPS by default which is not compatible with this test")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass
