# Copyright 2026 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
# reserved.
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
"""Testing suite for the PyTorch MusicFlamingo model."""

import json
import os
import unittest
from pathlib import Path

from transformers import (
    AudioFlamingo3EncoderConfig,
    AutoProcessor,
    MusicFlamingoConfig,
    MusicFlamingoForConditionalGeneration,
    Qwen2Config,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_deterministic_for_xpu,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch


class MusicFlamingoModelTester(ALMModelTester):
    """
    Builds a tiny MusicFlamingo config and synthetic inputs that respect MusicFlamingo's
    post-pool token accounting: num <sound> tokens per sample == post-pool frame count.
    """

    config_class = MusicFlamingoConfig
    conditional_generation_class = MusicFlamingoForConditionalGeneration
    text_config_class = Qwen2Config
    audio_config_class = AudioFlamingo3EncoderConfig
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        # feat_seq_length=60 → (60-1)//2+1=30 → (30-2)//2+1=15 audio embed tokens.
        kwargs.setdefault("feat_seq_length", 60)
        kwargs.setdefault("max_source_positions", (kwargs["feat_seq_length"] - 1) // 2 + 1)
        super().__init__(parent, **kwargs)

    def create_audio_mask(self):
        # Deterministic full-length mask — base default uses unseeded Python `random`, which makes
        # multi-call generation-comparison tests (e.g. assisted decoding vs greedy) flaky.
        return torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.bool).to(torch_device)

    def get_audio_embeds_mask(self, audio_mask):
        # AudioFlamingo3Encoder._get_feat_extract_output_lengths: conv2 (k=3,s=2) then avg_pool (k=2,s=2).
        input_lengths = audio_mask.sum(-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        max_len = int(output_lengths.max().item())
        positions = torch.arange(max_len, device=audio_mask.device)[None, :]
        return (positions < output_lengths[:, None]).long()

    def get_config(self):
        # MusicFlamingoConfig requires rope_parameters.
        config = super().get_config()
        config.rope_parameters = {"rope_type": "default", "rope_theta": 2048, "partial_rotary_factor": 0.5}
        return config


@require_torch
class MusicFlamingoForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `MusicFlamingoForConditionalGeneration`.
    """

    model_tester_class = MusicFlamingoModelTester
    pipeline_model_mapping = (
        {
            "text-to-speech": MusicFlamingoForConditionalGeneration,
            "audio-text-to-text": MusicFlamingoForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    def test_rotary_window_axis_resets_per_audio(self):
        config = self.model_tester.get_config()
        pos_emb = MusicFlamingoForConditionalGeneration(config).pos_emb.to(torch_device)

        timestamps = torch.tensor(
            [
                [0.00, 0.04, 0.08],
                [30.00, 30.04, 30.08],
                [60.00, 60.04, 60.08],
                [0.00, 0.04, 0.08],
                [30.00, 30.04, 30.08],
            ],
            device=torch_device,
        )
        cos, sin = pos_emb(timestamps, seq_len=timestamps.shape[1])

        torch.testing.assert_close(cos[0], cos[3])
        torch.testing.assert_close(sin[0], sin[3])
        torch.testing.assert_close(cos[1], cos[4])
        torch.testing.assert_close(sin[1], sin[4])
        self.assertFalse(torch.allclose(cos[0], cos[1]))

    def test_build_audio_timestamps_reconstructs_windows_from_input_ids(self):
        config = self.model_tester.get_config()
        model = MusicFlamingoForConditionalGeneration(config).to(torch_device).eval()
        num_windows = 5
        feat_seq_length = self.model_tester.feat_seq_length
        input_features_mask = torch.ones([num_windows, feat_seq_length], dtype=torch.bool, device=torch_device)
        input_ids = ids_tensor([2, 60], config.text_config.vocab_size - 2).to(torch_device) + 2
        input_ids[0, :45] = config.audio_token_id
        input_ids[1, :30] = config.audio_token_id

        _, post_lengths = model.audio_tower._get_feat_extract_output_lengths(
            input_features_mask.sum(-1).to(torch.long)
        )
        max_post_length = int(post_lengths.max().item())
        audio_embed_frame_step = config.audio_frame_step * 4
        frame_offsets = (
            torch.arange(max_post_length, dtype=torch.float32, device=torch_device) * audio_embed_frame_step
        )
        audio_timestamps = torch.stack(
            [
                0 * max_post_length * audio_embed_frame_step + frame_offsets,
                1 * max_post_length * audio_embed_frame_step + frame_offsets,
                2 * max_post_length * audio_embed_frame_step + frame_offsets,
                0 * max_post_length * audio_embed_frame_step + frame_offsets,
                1 * max_post_length * audio_embed_frame_step + frame_offsets,
            ]
        )

        inferred = model._build_audio_timestamps(input_ids, post_lengths, max_post_length)
        torch.testing.assert_close(inferred, audio_timestamps)

    @unittest.skip(
        reason="This test does not apply to MusicFlamingo since High-level inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass


@require_torch
class MusicFlamingoForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Original model is private, but expected outputs are computed with checkpoint/code during integration.
    """

    @classmethod
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = os.environ.get("MUSIC_FLAMINGO_TEST_CHECKPOINT", "nvidia/music-flamingo-2601-hf")
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)
        cls.max_new_tokens = 50

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_fixture_single_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/a3226a0ba25e51be84a4808a79b59257#file-reproducer_hf-py
        """
        path = Path(__file__).parent.parent.parent / "fixtures/musicflamingo/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_ids = torch.tensor(raw["token_ids"])
        exp_txt = raw["transcriptions"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3",
                    },
                ],
            }
        ]

        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map="auto", dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)
        seq = model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)

    @require_deterministic_for_xpu
    @slow
    def test_fixture_batched_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/a3226a0ba25e51be84a4808a79b59257#file-reproducer_hf-py
        """
        # fmt: off
        exp_ids = Expectations(
            {
                ("cuda", None): torch.tensor([
                    [1986, 3754, 374, 458, 94509, 19461, 98875, 55964, 3528, 1163, 681, 55964, 11598, 55564, 429, 57843, 279, 9842, 3040, 55964, 263, 55964, 1782, 55964, 30449, 27235, 315, 11416, 19461, 98875, 448, 279, 68897, 11, 10581, 52760, 42898, 975, 14260, 315, 6481, 97431, 55964, 13573, 2591, 2420, 13, 220, 576, 8090],
                    [334, 68043, 220, 16, 1019, 33648, 9287, 88828, 304, 51454, 11, 12711, 28347, 261, 304, 279, 3054, 11, 24353, 20783, 18707, 30789, 11, 22502, 4614, 389, 279, 49293, 271, 334, 68043, 220, 17, 1019, 26843, 2367, 98091, 389, 279, 39612, 11, 304, 17172, 582, 6950, 11, 14697, 41315, 311, 279],
                ]),
                ("xpu", None): torch.tensor([
                    [1986, 3754, 374, 458, 94509, 19461, 98875, 55964, 3528, 1163, 681, 55964, 11598, 55564, 429, 57843, 279, 9842, 3040, 55964, 263, 55964, 1782, 55964, 30449, 27235, 315, 11416, 19461, 98875, 448, 279, 68897, 11, 10581, 52760, 42898, 975, 14260, 315, 6481, 97431, 55964, 13573, 2591, 2420, 13, 220, 576, 8090],
                    [334, 68043, 220, 16, 1019, 33648, 9287, 88828, 304, 51454, 11, 12711, 28347, 261, 304, 279, 3054, 11, 24353, 20783, 18707, 30789, 11, 22502, 4614, 389, 2518, 49293, 271, 334, 68043, 220, 17, 1019, 26843, 2367, 98091, 389, 279, 39612, 11, 304, 17172, 582, 6950, 11, 14697, 41315, 311, 279],
                ]),
            }
        ).get_expectation()
        exp_txt = Expectations(
            {
                ("cuda", None): [
                    "This track is an uplifting Eurodance‑style Trance‑Pop anthem that blends the driving four‑on‑the‑floor pulse of classic Eurodance with the soaring, melodic synth work typical of modern trance‑infused pop.  The duration",
                    "**Verse 1**\nMidnight cravings in bloom, lights flicker in the room, pepperoni dreams arise, pizza party on the skies\n\n**Verse 2**\nCheese melts on the crust, in flavor we trust, boxes stacked to the",
                ],
                ("xpu", None): [
                    "This track is an uplifting Eurodance‑style Trance‑Pop anthem that blends the driving four‑on‑the‑floor pulse of classic Eurodance with the soaring, melodic synth work typical of modern trance‑infused pop.  The duration",
                    "**Verse 1**\nMidnight cravings in bloom, lights flicker in the room, pepperoni dreams arise, pizza party on red skies\n\n**Verse 2**\nCheese melts on the crust, in flavor we trust, boxes stacked to the",
                ],
            }
        ).get_expectation()
        # fmt: on

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3",
                        },
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate a structured lyric sheet from the input music.",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_2.mp3",
                        },
                    ],
                }
            ],
        ]

        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map="auto", dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)
        seq = model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False)
        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)
