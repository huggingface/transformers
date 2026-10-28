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

import tempfile
import unittest
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import MiMoV2Processor
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, require_vision
from transformers.utils import is_torch_available
from transformers.video_utils import VideoMetadata

from ...test_processing_common import ProcessorTesterMixin
from .test_processing_mimo_v2 import make_processor


if is_torch_available():
    import torch


@require_torch
@require_torchcodec
@require_torchvision
@require_vision
class MiMoV2ProcessorCommonTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = MiMoV2Processor

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": (
                "{% for message in messages %}"
                "{% if message['content'] is string %}{{ message['content'] }}"
                "{% else %}{% for item in message['content'] %}"
                "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
                "{% elif item['type'] == 'audio' %}<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
                "{% elif item['type'] == 'video' %}<|vision_start|><|video_pad|><|vision_end|>"
                "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"
            )
        }

    @classmethod
    def _setup_from_components(cls):
        cls.video_path = hf_hub_download(
            "hf-internal-testing/test-videos", "tiny_video_320x240.mp4", repo_type="dataset"
        )
        return make_processor(**cls.prepare_processor_dict())

    def prepare_videos_inputs(self, batch_size=None):
        # File decoding supplies the source timestamps required by MiMo's video placeholders.
        return self.video_path if batch_size is None else [self.video_path] * batch_size

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 96},
            {"num_frames": None, "fps": 10, "expected_dim": 0, "output_length": 96},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 144},
            {"expected_dim": 0, "output_length": 48},
        ]

    def test_apply_chat_template_video_frame_sampling(self):
        from PIL import Image

        processor = self.get_processor()
        messages = [{"role": "user", "content": [{"type": "video", "path": self.video_path}]}]
        kwargs = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
        for options in self.video_sampling_expectations:
            expected_length = options.pop("output_length")
            expected_dim = options.pop("expected_dim")
            output = processor.apply_chat_template(messages, processor_kwargs=options, **kwargs)
            self.assertEqual(output.pixel_values_videos.shape[expected_dim], expected_length)

        with tempfile.TemporaryDirectory() as directory:
            frame_path = str(Path(directory, "frame.png"))
            Image.new("RGB", (8, 8)).save(frame_path)
            messages[0]["content"][0] = {"type": "video", "url": [frame_path, frame_path]}
            # Unlike a video file, frame paths do not carry their original source timestamps.
            options = {"video_metadata": VideoMetadata(total_num_frames=2, fps=2, frames_indices=[0, 1])}
            output = processor.apply_chat_template(
                messages, processor_kwargs={**options, "do_sample_frames": False}, **kwargs
            )
            self.assertEqual(output.pixel_values_videos.shape, (16, 24))
            with self.assertRaisesRegex(ValueError, "Sampling frames from a list of images is not supported"):
                processor.apply_chat_template(
                    messages, processor_kwargs={**options, "do_sample_frames": True}, **kwargs
                )

    @parameterized.expand([(1, "pt")])
    def test_apply_chat_template_decoded_video(self, batch_size, return_tensors):
        processor = self.get_processor()
        frames, metadata = processor.video_processor.fetch_videos([self.video_path])
        messages = [{"role": "user", "content": [{"type": "video", "video": frames[0]}]}]
        kwargs = {"tokenize": True, "return_dict": True, "return_tensors": return_tensors}
        actual = processor.apply_chat_template(
            messages, processor_kwargs={"video_metadata": list(metadata), "do_sample_frames": False}, **kwargs
        )
        messages[0]["content"][0] = {"type": "video", "path": self.video_path}
        expected = processor.apply_chat_template(messages, processor_kwargs={"do_sample_frames": False}, **kwargs)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities=["image", "video", "audio"]),
            images=self.prepare_images_inputs(),
            videos=self.prepare_videos_inputs(),
            audio=self.prepare_audio_inputs(),
            return_tensors="pt",
        )
        # Codec IDs are an alternative to waveform features; modality IDs are opt-in.
        self.assertSetEqual(set(inputs), set(processor.model_input_names) - {"input_audio_codes", "mm_token_type_ids"})
        codec_inputs = processor(
            text=processor.audio_token,
            audio=np.zeros((8, processor.feature_extractor.audio_channels), dtype=np.int64),
            is_tokenized=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )
        self.assertIn("input_audio_codes", codec_inputs)
        self.assertIn("mm_token_type_ids", codec_inputs)
        self.assertNotIn("input_features", codec_inputs)
        self.assertTrue(set(codec_inputs) <= set(processor.model_input_names))

    def test_subprocessor_defaults_3_audio(self):
        processor = self.get_processor()
        audio = self.prepare_audio_inputs()
        expected = processor.feature_extractor(audio, return_tensors="pt")
        actual = processor(audio=audio, return_tensors="pt")
        for key in expected:
            processor_key = "feature_attention_mask" if key == "attention_mask" else key
            torch.testing.assert_close(actual[processor_key], expected[key], rtol=0, atol=0)
