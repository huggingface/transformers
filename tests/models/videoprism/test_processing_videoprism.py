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

import torch

from transformers.testing_utils import require_tokenizers, require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import LlavaOnevisionVideoProcessor, VideoPrismProcessor, VideoPrismTokenizer

TENNIS_VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"
INTEGRATION_NUM_FRAMES = 16
INTEGRATION_FRAME_SIZE = 288
VIDEO_PRISM_LVT_CHECKPOINT = "MHRDYN7/videoprism-lvt-base-f16r288"


def prepare_tennis_frames():
    tennis_video = url_to_local_path(TENNIS_VIDEO_URL)
    video_processor = LlavaOnevisionVideoProcessor(
        size={"height": INTEGRATION_FRAME_SIZE, "width": INTEGRATION_FRAME_SIZE},
        do_normalize=False,
    )
    return tennis_video, video_processor(
        videos=tennis_video,
        return_tensors="pt",
        do_sample_frames=True,
        num_frames=INTEGRATION_NUM_FRAMES,
    )["pixel_values_videos"]


@require_tokenizers
@require_vision
@require_torch
class VideoPrismProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VideoPrismProcessor
    video_text_kwargs_max_length = 64

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tennis_video, cls.tennis_frames = prepare_tennis_frames()

    @classmethod
    def _setup_tokenizer(cls):
        return VideoPrismTokenizer.from_pretrained(VIDEO_PRISM_LVT_CHECKPOINT)

    @classmethod
    def _setup_video_processor(cls):
        return LlavaOnevisionVideoProcessor(
            size={"height": INTEGRATION_FRAME_SIZE, "width": INTEGRATION_FRAME_SIZE},
            do_normalize=False,
        )

    def test_processor_video_tennis_video(self):
        """VideoPrismProcessor on tennis.mp4 matches prepare_tennis_frames golden pixels."""
        processor = self.processor_class(
            video_processor=self._setup_video_processor(),
            tokenizer=self._setup_tokenizer(),
        )
        processor_out = processor(
            videos=self.tennis_video,
            return_tensors="pt",
            do_sample_frames=True,
            num_frames=INTEGRATION_NUM_FRAMES,
        )
        pixel_values_videos = processor_out["pixel_values_videos"]
        self.assertEqual(pixel_values_videos.shape[1], INTEGRATION_NUM_FRAMES)
        self.assertEqual(pixel_values_videos.shape[-2:], (INTEGRATION_FRAME_SIZE, INTEGRATION_FRAME_SIZE))

        torch.testing.assert_close(pixel_values_videos, self.tennis_frames, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            pixel_values_videos[0, 0, 0, 144:147, 144:147],
            self.tennis_frames[0, 0, 0, 144:147, 144:147],
            rtol=1e-4,
            atol=1e-4,
        )
