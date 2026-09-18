# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from transformers import Glm46VProcessor


@require_vision
@require_torch
class Glm46VProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Glm46VProcessor
    # Tiny processor created with make_tiny_processor.py from "THUDM/GLM-4.1V-9B-Thinking"
    tiny_model_id = "hf-internal-testing/tiny-processor-glm4v"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        return super()._setup_from_pretrained(
            model_id,
            do_sample_frames=False,
            patch_size=4,
            size={"shortest_edge": 12 * 12, "longest_edge": 18 * 18},
            **kwargs,
        )

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 4},
            {"num_frames": None, "fps": 16, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
        ]

    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video"])
        image_input = self.prepare_images_inputs()
        video_inputs = self.prepare_videos_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs}
        inputs = processor(**inputs_dict, return_tensors="pt", do_sample_frames=False)

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @parameterized.expand([(1, "pt")])
    @unittest.skip("Mode requires metadata to be always passed by users")
    def test_apply_chat_template_decoded_video(self, batch_size: int, return_tensors: str):
        pass
