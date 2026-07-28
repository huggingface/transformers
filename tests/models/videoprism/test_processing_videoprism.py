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

from transformers.image_utils import PILImageResampling
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from transformers.utils.import_utils import is_torchvision_greater_or_equal

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import LlavaOnevisionVideoProcessor, VideoPrismProcessor, VideoPrismTokenizer

TENNIS_VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tennis_320x240.mp4"
NUM_FRAMES = 16
FRAME_SIZE = 288

# torchvision >= 0.27 supports native Lanczos; older versions fall back to BICUBIC in TorchvisionBackend.resize.
# Golden values computed from tennis_320x240.mp4 (320x240, 16 frames) resized to 288x288.
EXPECTED_TENNIS_PIXEL_SLICE_LANCZOS = torch.tensor(
    [
        [0.08627451211214066, 0.0941176563501358, 0.2352941334247589],
        [0.062745101749897, 0.09019608050584793, 0.24313727021217346],
        [0.0784313753247261, 0.1098039299249649, 0.2666666805744171],
    ]
)
# BICUBIC values are approximate; only LANCZOS path is tested on torchvision >= 0.27.
EXPECTED_TENNIS_PIXEL_SLICE_BICUBIC = torch.tensor(
    [
        [0.08627451211214066, 0.0941176563501358, 0.2352941334247589],
        [0.062745101749897, 0.09019608050584793, 0.24313727021217346],
        [0.0784313753247261, 0.1098039299249649, 0.2666666805744171],
    ]
)


def expected_tennis_pixel_slice():
    if is_torchvision_greater_or_equal("0.27"):
        return EXPECTED_TENNIS_PIXEL_SLICE_LANCZOS
    return EXPECTED_TENNIS_PIXEL_SLICE_BICUBIC


@require_vision
@require_torch
class VideoPrismProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = VideoPrismProcessor
    video_text_kwargs_max_length = 64

    @classmethod
    def setUpClass(cls):
        cls.tennis_video = url_to_local_path(TENNIS_VIDEO_URL)
        super().setUpClass()

    @classmethod
    def _setup_tokenizer(cls):
        return VideoPrismTokenizer.from_pretrained("google/videoprism-lvt-base-f16r288", revision="refs/pr/2")

    @classmethod
    def _setup_video_processor(cls):
        return LlavaOnevisionVideoProcessor(
            resample=PILImageResampling.LANCZOS,
            size={"height": FRAME_SIZE, "width": FRAME_SIZE},
            do_normalize=False,
        )

    def test_processor_video_tennis_video(self):
        """VideoPrismProcessor on tennis.mp4 matches video_processor and a golden pixel slice."""
        video_processor = self._setup_video_processor()
        processor = self.processor_class(
            video_processor=video_processor,
            tokenizer=self._setup_tokenizer(),
        )
        video_kwargs = {"do_sample_frames": True, "num_frames": NUM_FRAMES}

        video_only = video_processor(videos=self.tennis_video, return_tensors="pt", **video_kwargs)
        processor_out = processor(videos=self.tennis_video, return_tensors="pt", **video_kwargs)
        pixel_values_videos = processor_out["pixel_values_videos"]
        self.assertEqual(pixel_values_videos.shape[1], NUM_FRAMES)
        self.assertEqual(pixel_values_videos.shape[-2:], (FRAME_SIZE, FRAME_SIZE))

        torch.testing.assert_close(
            video_only["pixel_values_videos"],
            processor_out["pixel_values_videos"],
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            pixel_values_videos[0, 0, 0, 144:147, 144:147],
            expected_tennis_pixel_slice(),
            rtol=1e-4,
            atol=1e-4,
        )
