# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from transformers import (
    AudioVisualFlamingoConfig,
    AudioVisualFlamingoProcessor,
    AutoTokenizer,
    SiglipImageProcessor,
    WhisperFeatureExtractor,
)
from transformers.models.audiovisualflamingo.processing_audiovisualflamingo import _load_audio_hf_with_info
from transformers.testing_utils import require_torch, require_vision


MEDIA_TOKENS = AudioVisualFlamingoConfig.media_tokens
MM_BOS_EOS_TOKENS = AudioVisualFlamingoConfig.mm_bos_eos_tokens


def _make_audio(seconds: float, sampling_rate: int = 16_000, frequency: float = 220.0) -> np.ndarray:
    steps = int(seconds * sampling_rate)
    timeline = np.linspace(0.0, seconds, steps, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * frequency * timeline).astype(np.float32)


@require_torch
@require_vision
class AudioVisualFlamingoProcessorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", use_fast=True)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    *MEDIA_TOKENS.values(),
                    *(token for bos_eos_tokens in MM_BOS_EOS_TOKENS.values() for token in bos_eos_tokens),
                ]
            }
        )

        processor = AudioVisualFlamingoProcessor(
            image_processor=SiglipImageProcessor(
                crop_size={"height": 384, "width": 384},
                size={"height": 384, "width": 384},
            ),
            feature_extractor=WhisperFeatureExtractor(
                feature_size=128,
                chunk_length=30,
                sampling_rate=16_000,
                hop_length=60,
            ),
            tokenizer=tokenizer,
            image_aspect_ratio="dynamic_s2",
            s2_scales=[384, 768, 1152],
            num_video_frames=8,
            padding_side="left",
        )

        cls.tmpdirname = tempfile.mkdtemp()
        processor.save_pretrained(cls.tmpdirname)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def get_processor(self, **kwargs) -> AudioVisualFlamingoProcessor:
        return AudioVisualFlamingoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    def test_apply_chat_template_batched_audio_groups_flat_inputs(self):
        processor = self.get_processor()

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": _make_audio(0.5)},
                        {"type": "audio", "audio": _make_audio(0.75, frequency=330.0)},
                        {"type": "text", "text": "Compare these clips."},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": _make_audio(0.6, frequency=440.0)},
                        {"type": "text", "text": "Describe this clip."},
                    ],
                }
            ],
        ]

        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
        )

        self.assertEqual(len(inputs["media"]["sound"]), 3)
        self.assertEqual([len(sample) for sample in inputs["media"]["audio_info"]], [2, 1])
        self.assertEqual(inputs["attention_mask"].dtype, torch.bool)

    def test_dynamic_s2_block_sizes_are_aggregated_per_sample(self):
        processor = self.get_processor()

        outputs = processor(
            text=[
                f"{processor.image_token} Describe the first image.",
                f"{processor.image_token} Describe the second image.",
            ],
            images=[
                [Image.new("RGB", (640, 320), color="red")],
                [Image.new("RGB", (320, 640), color="blue")],
            ],
        )

        self.assertEqual(len(outputs["media_config"]["image"]["block_sizes"]), 2)
        self.assertEqual((outputs["input_ids"] == processor.image_token_id).sum().item(), 2)

    def test_video_audio_placeholder_is_inserted_from_video_loader_output(self):
        processor = self.get_processor()
        dummy_frame = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), mode="RGB")

        def fake_extract_video(video_input, config):
            del config
            audio_info = {
                "audio_start_sec": 0.0,
                "audio_end_sample_sec": 1.0,
                "ori_audio_duration": 1.0,
            }
            video_info = {
                "video_path": str(video_input),
                "has_audio": True,
                "video_duration": 1.0,
                "audio_info": audio_info,
                "video_frame_times": [0.0],
            }
            return [dummy_frame], _make_audio(1.0), video_info

        with patch(
            "transformers.models.audiovisualflamingo.processing_audiovisualflamingo._extract_video_hf",
            side_effect=fake_extract_video,
        ):
            inputs = processor(
                text=[f"{processor.video_token} Summarize the clip."],
                videos=[["dummy-video.mp4"]],
            )

        self.assertEqual(len(inputs["media"]["sound"]), 1)
        self.assertEqual([len(sample) for sample in inputs["media"]["audio_info"]], [1])
        self.assertEqual((inputs["input_ids"] == processor.sound_token_id).sum().item(), 1)

    def test_audio_loader_falls_back_to_pyav_for_media_containers(self):
        runtime_config = SimpleNamespace(audio_sampling_rate=16_000, audio_chunk_length=120, random_audio_sample=False)

        with (
            patch(
                "transformers.models.audiovisualflamingo.processing_audiovisualflamingo.load_audio",
                side_effect=RuntimeError("decode failed"),
            ) as mocked_load_audio,
            patch(
                "transformers.models.audiovisualflamingo.processing_audiovisualflamingo._load_audio_track_with_pyav",
                return_value=_make_audio(1.0),
            ) as mocked_fallback,
        ):
            waveform, audio_info = _load_audio_hf_with_info("dummy-video.mp4", runtime_config)

        mocked_load_audio.assert_called_once_with("dummy-video.mp4", sampling_rate=16_000)
        mocked_fallback.assert_called_once_with("dummy-video.mp4", 16_000)
        self.assertEqual(waveform.shape[0], audio_info["new_audio_n_samples"])
        self.assertEqual(audio_info["new_audio_chunk_length"], 30)

    def test_model_input_names_include_media_keys(self):
        processor = self.get_processor()
        self.assertIn("media", processor.model_input_names)
        self.assertIn("media_config", processor.model_input_names)

    def test_standard_component_configs_resolve_to_subconfigs(self):
        config = AudioVisualFlamingoConfig(
            text_config={
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "vocab_size": 256,
            },
            vision_config={
                "model_type": "siglip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 384,
                "patch_size": 14,
            },
            audio_config={
                "model_type": "qwen2_audio_encoder",
                "num_mel_bins": 128,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 64,
                "d_model": 32,
            },
        )

        self.assertEqual(config.text_config.model_type, "qwen2")
        self.assertEqual(config.vision_config.model_type, "siglip_vision_model")
        self.assertEqual(config.audio_config.model_type, "qwen2_audio_encoder")

    def test_config_keeps_only_canonical_runtime_fields(self):
        config = AudioVisualFlamingoConfig(
            s2_scales=[448, 896, 1344],
            image_encoder={"_target_": "BasicImageEncoder"},
            video_encoder={"_target_": "TSPVideoEncoder", "embed_time": "True"},
            sound_encoder={"_target_": "BasicSoundEncoder", "embed_time": "True"},
        )

        self.assertEqual(config.s2_scales, [448, 896, 1344])
        self.assertEqual(config.image_encoder["_target_"], "BasicImageEncoder")
        self.assertEqual(config.video_encoder["_target_"], "TSPVideoEncoder")
        self.assertEqual(config.sound_encoder["_target_"], "BasicSoundEncoder")

        config_dict = config.to_dict()
        self.assertNotIn("audio_sampling_rate", config_dict)
        self.assertNotIn("audio_chunk_length", config_dict)
        self.assertNotIn("audio_hop_length", config_dict)
        self.assertNotIn("num_video_frames", config_dict)
        self.assertNotIn("max_tiles", config_dict)
