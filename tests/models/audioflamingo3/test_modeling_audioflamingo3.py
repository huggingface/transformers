# Copyright 2025 NVIDIA CORPORATION and the HuggingFace Inc. team. All rights
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
"""Testing suite for the PyTorch AudioFlamingo3 model."""

import json
import unittest
from pathlib import Path

import pytest

from transformers import (
    AudioFlamingo3Config,
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester


if is_torch_available():
    import torch


class AudioFlamingo3ModelTester(ALMModelTester):
    config_class = AudioFlamingo3Config
    conditional_generation_class = AudioFlamingo3ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault(
            "audio_config",
            {
                "model_type": "audioflamingo3_encoder",
                "hidden_size": 16,
                "num_attention_heads": 4,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_mel_bins": 80,
                "max_source_positions": 30,
                "initializer_range": 0.02,
            },
        )
        super().__init__(parent, **kwargs)

    def get_audio_mask_key(self):
        return "input_features_mask"

    def create_audio_mask(self, audio_features):
        return torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.bool).to(torch_device)


@require_torch
class AudioFlamingo3ForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `AudioFlamingo3ForConditionalGeneration`.
    """

    model_tester_class = AudioFlamingo3ModelTester
    # TODO: @eustlb, this is incorrect
    pipeline_model_mapping = (
        {
            "text-to-speech": AudioFlamingo3ForConditionalGeneration,
            "audio-text-to-text": AudioFlamingo3ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    @unittest.skip(
        reason="This test does not apply to AudioFlamingo3 since inputs_embeds corresponding to audio tokens "
        "are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Compile not yet supported for AudioFlamingo3 models")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported for AudioFlamingo3 models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="AudioFlamingo3 tests avoid right-padding equivalence; fusion is in-place.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass


@require_torch
class AudioFlamingo3ForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Slow tests against the public checkpoint to validate processor-model alignment and in-place fusion.
    """

    @classmethod
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "nvidia/audio-flamingo-3-hf"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_fixture_single_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/c979f0f1a2b9223fa137faf1c02022d4#file-reproducer-py
        """
        path = Path(__file__).parent.parent.parent / "fixtures/audioflamingo3/expected_results_single.json"
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
                        "text": "What is surprising about the relationship between the barking and the music?",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
                    },
                ],
            }
        ]

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)
        seq = model.generate(**batch)
        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)

    @slow
    def test_fixture_batched_matches(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/c979f0f1a2b9223fa137faf1c02022d4#file-reproducer-py
        """
        path = Path(__file__).parent.parent.parent / "fixtures/audioflamingo3/expected_results_batched.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        exp_ids = torch.tensor(raw["token_ids"])
        exp_txt = raw["transcriptions"]

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is surprising about the relationship between the barking and the music?",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/dogs_barking_in_sync_with_the_music.wav",
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
                            "text": "Why is the philosopher's name mentioned in the lyrics? "
                            "(A) To express a sense of nostalgia "
                            "(B) To indicate that language cannot express clearly, satirizing the inversion of black and white in the world "
                            "(C) To add depth and complexity to the lyrics "
                            "(D) To showcase the wisdom and influence of the philosopher",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/Ch6Ae9DT6Ko_00-04-03_00-04-31.wav",
                        },
                    ],
                }
            ],
        ]

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        ).eval()

        batch = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device, dtype=model.dtype)
        seq = model.generate(**batch)
        inp_len = batch["input_ids"].shape[1]
        gen_ids = seq[:, inp_len:] if seq.shape[1] >= inp_len else seq

        torch.testing.assert_close(gen_ids.cpu(), exp_ids)
        txt = self.processor.decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)
