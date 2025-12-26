# coding=utf-8
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
import tempfile
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

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class AudioFlamingo3ModelTester:
    """
    Builds a tiny AudioFlamingo3 config and synthetic inputs that respect AF3's
    post-pool token accounting: num <sound> tokens per sample == post-pool frame count.
    """

    def __init__(
        self,
        parent,
        audio_token_id=0,
        seq_length=25,
        feat_seq_length=60,
        text_config=None,
        audio_config=None,
        is_training=True,
    ):
        self.parent = parent
        self.audio_token_id = audio_token_id
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length
        self.is_training = is_training

        # Small text backbone (Qwen2-ish)
        if text_config is None:
            text_config = {
                "model_type": "qwen2",
                "intermediate_size": 36,
                "initializer_range": 0.02,
                "hidden_size": 32,
                "max_position_embeddings": 52,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "use_labels": True,
                "use_mrope": False,
                "vocab_size": 99,
                "pad_token_id": 1,  # Ensure pad token != audio token
            }
        # Small audio encoder (AF3 Whisper-style)
        if audio_config is None:
            audio_config = {
                "model_type": "audioflamingo3_encoder",
                "hidden_size": 16,
                "num_attention_heads": 4,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_mel_bins": 80,
                "max_source_positions": 30,
                "initializer_range": 0.02,
            }

        self.text_config = text_config
        self.audio_config = audio_config

        self.batch_size = 3
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return AudioFlamingo3Config(
            text_config=self.text_config,
            audio_config=self.audio_config,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        # (#windows == batch_size, n_mels, T_mel)
        input_features_values = floats_tensor(
            [self.batch_size, self.audio_config["num_mel_bins"], self.feat_seq_length]
        )
        config = self.get_config()
        # Per-window mel validity (all ones => full length)
        input_features_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.bool).to(torch_device)
        return config, input_features_values, input_features_mask

    def _post_pool_tokens_per_window(self, T_mel):
        # Mirror AF3 processor math:
        pre = (T_mel - 1) // 2 + 1
        post = (pre - 2) // 2 + 1
        return post

    def prepare_config_and_inputs_for_common(self):
        config, input_features_values, input_features_mask = self.prepare_config_and_inputs()
        # Every window has same T_mel here
        num_audio_tokens_per_sample = self._post_pool_tokens_per_window(input_features_values.shape[-1])

        # Build token ids with valid range and K <sound> tokens
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=torch_device)
        attention_mask[:, :1] = 0  # left padding sentinel

        # Fill first K positions (after padding) with the audio token id, for each sample
        input_ids[:, 1 : 1 + num_audio_tokens_per_sample] = config.audio_token_id

        inputs_dict = {
            "input_features": input_features_values,
            "input_features_mask": input_features_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class AudioFlamingo3ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `AudioFlamingo3ForConditionalGeneration`.
    """

    all_model_classes = (AudioFlamingo3ForConditionalGeneration,) if is_torch_available() else ()
    # TODO: @eustlb, this is incorrect
    pipeline_model_mapping = (
        {
            "text-to-speech": AudioFlamingo3ForConditionalGeneration,
            "audio-text-to-text": AudioFlamingo3ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True

    def setUp(self):
        self.model_tester = AudioFlamingo3ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=AudioFlamingo3Config, has_text_modality=False)

    @unittest.skip(
        reason="This test does not apply to AudioFlamingo3 since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
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

    @unittest.skip(reason="AudioFlamingo3 has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    def test_sdpa_can_dispatch_composite_models(self):
        # AF3 is audio+text composite; verify SDPA toggles propagate to submodules.
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # SDPA (default)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                audio_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == audio_attn)

                # Eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for _, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


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
                        "text": "Transcribe the input speech.",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/Why_do_we_ask_questions_converted.wav",
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
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
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
        txt = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        self.assertListEqual(txt, exp_txt)
