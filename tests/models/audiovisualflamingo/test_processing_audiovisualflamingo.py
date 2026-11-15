# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights reserved.
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

import numpy as np

from transformers import (
    AudioVisualFlamingoImageProcessor,
    AudioVisualFlamingoProcessor,
    AudioVisualFlamingoVideoProcessor,
    AutoTokenizer,
    WhisperFeatureExtractor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, require_vision
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch


@require_vision
@require_torchvision
@require_torch
class AudioVisualFlamingoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = AudioVisualFlamingoProcessor
    checkpoint = "nvidia/nemotron-labs-audio-visual-flamingo-hf"
    # An audio window alone is encoded into 750 tokens
    audio_text_kwargs_max_length = 800
    audio_unstructured_max_length = 800

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained(cls.checkpoint)

    @classmethod
    def _setup_image_processor(cls):
        # Images are not tiled by default, tiling is tested in `test_image_tiling`
        return AudioVisualFlamingoImageProcessor(size={"height": 28, "width": 28}, s2_scales=[28], max_tiles=1)

    @classmethod
    def _setup_video_processor(cls):
        return AudioVisualFlamingoVideoProcessor(size={"height": 28, "width": 28}, num_frames=8)

    @classmethod
    def _setup_feature_extractor(cls):
        return WhisperFeatureExtractor(feature_size=128)

    @staticmethod
    def prepare_processor_dict():
        from transformers.models.audiovisualflamingo.convert_audiovisualflamingo_to_hf import CHAT_TEMPLATE

        return {"chat_template": CHAT_TEMPLATE, "image_seq_length": 4}

    @property
    def video_sampling_expectations(self):
        # Videos are sampled to `num_frames` frames (8 by default), sampling by `fps` is not supported
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 1, "output_length": 3},
            {"do_sample_frames": False, "fps": 10, "expected_dim": 1, "output_length": 11},
            {"do_sample_frames": False, "expected_dim": 1, "output_length": 11},
            {"expected_dim": 1, "output_length": 8},
        ]

    def decode_layout(self, input_ids):
        """Shortens the runs of placeholder tokens of a decoded prompt."""
        processor = self.get_processor()
        text, layout = processor.tokenizer.decode(input_ids), []
        for token in (processor.image_token, processor.video_token, processor.audio_token):
            text = text.replace(token, f"\0{token}\0")
        for part in text.split("\0"):
            if layout and part and part == layout[-1][0]:
                layout[-1][1] += 1
            elif part:
                layout.append([part, 1])
        return "".join(f"{part}x{count}" if count > 1 else part for part, count in layout)

    def test_image_tiling(self):
        processor = self.get_processor()
        image = np.zeros((40, 100, 3), dtype=np.uint8)

        # A single image is tiled at every S2 scale: 1 + 4 tiles, plus the 2x5 grid closest to its aspect ratio
        tiling = {"s2_scales": [28, 56, 84], "max_tiles": 12}
        inputs = processor(text="<image>Describe.", images=image, **tiling)
        self.assertEqual(inputs["pixel_values"].shape, (15, 3, 28, 28))
        self.assertEqual(inputs["image_grid_hw"].tolist(), [[2, 5]])
        self.assertEqual(self.decode_layout(inputs["input_ids"][0]), "<|image_bos|><image>x40\n<|image_eos|>Describe.")

        # Images of multi-image prompts are single tiles, and the image span is wrapped once
        inputs = processor(text="<image>vs<image>", images=[image, image], **tiling)
        self.assertEqual(inputs["pixel_values"].shape, (2, 3, 28, 28))
        self.assertEqual(inputs["image_grid_hw"].tolist(), [[1, 1], [1, 1]])
        self.assertEqual(
            self.decode_layout(inputs["input_ids"][0]), "<|image_bos|><image>x4\nvs<image>x4\n<|image_eos|>"
        )

    def test_audio_windows(self):
        processor = self.get_processor()
        # 40 seconds are padded to two 30 second windows of 3000 mel frames, encoded into 750 tokens each
        inputs = processor(text="<sound>Describe.", audio=np.zeros(40 * 16000, dtype=np.float32))
        self.assertEqual(inputs["input_features"].shape, (1, 128, 6000))
        self.assertTrue(inputs["input_features_mask"].all())
        self.assertEqual(self.decode_layout(inputs["input_ids"][0]), "<|sound_bos|><sound>x1500<|sound_eos|>Describe.")

    def test_video_with_audio_interleaving(self):
        processor = self.get_processor()
        # 40 seconds at 1 fps, sampled to 8 frames pooled into a single group of 4 tokens. The audio lasts 35 s.
        video = np.zeros((40, 28, 28, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=40, fps=1.0)
        audio = np.zeros(35 * 16000, dtype=np.float32)
        inputs = processor(
            text="<vila/video>Describe.",
            videos=[video],
            video_metadata=[metadata],
            audio=[audio],
            audio_from_video_indices=[0],
            load_audio_from_video=True,
        )
        self.assertEqual(inputs["pixel_values_videos"].shape, (1, 8, 3, 28, 28))
        self.assertEqual(inputs["video_timestamps"], [[0.0, 6.0, 11.0, 17.0, 22.0, 28.0, 33.0, 39.0]])
        # Two 30 s segments: 6 frames and 30 s of audio, then 2 frames and 5 s of audio
        self.assertEqual(
            self.decode_layout(inputs["input_ids"][0]),
            "<|sound_bos|><|sound_eos|><|video_bos|><vila/video>x3\n<sound>x750\n<vila/video>\n<sound>x125\n"
            "<|video_eos|>Describe.",
        )
        # Only the audio embeddings placed in the video are kept
        self.assertEqual(inputs["input_features_mask"].sum().item(), 4 * 875)

    def test_video_without_audio(self):
        processor = self.get_processor()
        inputs = processor(
            text="<vila/video>Describe.",
            videos=[np.zeros((16, 28, 28, 3), dtype=np.uint8)],
        )
        self.assertEqual(
            self.decode_layout(inputs["input_ids"][0]), "<|video_bos|><vila/video>x4<|video_eos|>Describe."
        )
        self.assertNotIn("input_features", inputs)

    def test_video_timestamps_keep_precision(self):
        processor = self.get_processor()
        video = np.zeros((40, 28, 28, 3), dtype=np.uint8)
        inputs = processor(
            text="<vila/video>", videos=[video], video_metadata=[VideoMetadata(total_num_frames=40, fps=29.97)]
        )

        # Frame times are Python floats, which casting the inputs to a lower precision leaves untouched
        expected = [[index / 29.97 for index in (0, 6, 11, 17, 22, 28, 33, 39)]]
        self.assertEqual(inputs.to(torch.bfloat16)["video_timestamps"], expected)

    @require_torchcodec
    def test_chat_template_audio_from_video(self):
        processor = self.get_processor()
        url = "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/sample_demo_1_320x240.mp4"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": url_to_local_path(url)},
                    {"type": "text", "text": "What is it?"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True, return_dict=True, load_audio_from_video=True
        )
        self.assertEqual(inputs["pixel_values_videos"].shape, (1, 8, 3, 28, 28))
        self.assertEqual(len(inputs["input_features"]), 1)
        # The audio track is interleaved with the frames, in a single segment for this short video
        layout = self.decode_layout(inputs["input_ids"][0])
        self.assertRegex(
            layout, r"<\|sound_bos\|><\|sound_eos\|><\|video_bos\|><vila/video>x4\n<sound>x\d+\n<\|video_eos\|>"
        )
