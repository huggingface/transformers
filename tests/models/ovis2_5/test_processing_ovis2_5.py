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

from transformers import Ovis2_5Processor
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, require_vision
from transformers.utils import is_torchcodec_usable

from ...test_processing_common import ProcessorTesterMixin


@require_vision
@require_torch
@require_torchvision
class Ovis2_5ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Ovis2_5Processor
    model_id = "ovis-community/Ovis2.5-2B"
    images_text_kwargs_max_length = 256
    images_unstructured_max_length = 256
    videos_text_kwargs_max_length = 2048
    videos_unstructured_max_length = 2048

    def _test_apply_chat_template(self, modality, batch_size, return_tensors, input_name, processor_name, input_data):
        if modality == "video" and isinstance(input_data[0], str) and not is_torchcodec_usable(decode="video"):
            self.skipTest("Video URL decoding requires Torchcodec")
        super()._test_apply_chat_template(modality, batch_size, return_tensors, input_name, processor_name, input_data)

    @property
    def video_sampling_expectations(self):
        # Each 240x320 fixture frame resizes to 416x544, yielding 26x34 patches.
        # The saved num_frames=8 also applies when num_frames is None or omitted.
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 3 * 884},
            {"num_frames": None, "fps": 10, "expected_dim": 0, "output_length": 8 * 884},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 0, "output_length": 11 * 884},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 11 * 884},
            {"expected_dim": 0, "output_length": 8 * 884},
        ]

    @require_torchcodec
    def test_apply_chat_template_video_frame_sampling(self):
        super().test_apply_chat_template_video_frame_sampling()
