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
"""Testing suite for the PyTorch moss_transcribe_diarize model."""

import json
import unittest
from pathlib import Path

from transformers import (
    AutoProcessor,
    MossTranscribeDiarizeConfig,
    MossTranscribeDiarizeForConditionalGeneration,
    MossTranscribeDiarizeModel,
    Qwen3Config,
    WhisperConfig,
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


class MossTranscribeDiarizeModelTester(ALMModelTester):
    config_class = MossTranscribeDiarizeConfig
    base_model_class = MossTranscribeDiarizeModel
    conditional_generation_class = MossTranscribeDiarizeForConditionalGeneration
    text_config_class = Qwen3Config
    audio_config_class = WhisperConfig
    audio_mask_key = None

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("feat_seq_length", 128)
        kwargs.setdefault("max_source_positions", (kwargs["feat_seq_length"] - 1) // 2 + 1)
        kwargs.setdefault("d_model", 16)
        kwargs.setdefault("hidden_size", 16)
        kwargs.setdefault("intermediate_size", 32)
        kwargs.setdefault("encoder_layers", 1)
        kwargs.setdefault("encoder_attention_heads", 2)
        kwargs.setdefault("encoder_ffn_dim", 32)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("audio_merge_size", 4)
        kwargs.setdefault("audio_token_id", 0)
        super().__init__(parent, **kwargs)

    def _prepare_modality_inputs(self, input_ids, config):
        num_audio_tokens = torch.full((self.batch_size,), 8, dtype=torch.long, device=torch_device)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        # 64 input frames -> 32 post-conv -> 8 merged tokens (merge_size=4), matching
        # `WhisperEncoder._get_feat_extract_output_lengths` + merge trim in `get_audio_features`.
        valid_mel_frames = 64
        input_features_mask = torch.zeros(self.batch_size, self.feat_seq_length, dtype=torch.long, device=torch_device)
        input_features_mask[:, :valid_mel_frames] = 1
        modality_inputs = {
            "input_features": self.create_audio_features(),
            "input_features_mask": input_features_mask,
            # 1 sample = 1 chunk, like the `torch.arange` mapping this replaces.
            "padding_mask": torch.ones(self.batch_size, 1, dtype=torch.long, device=torch_device),
        }
        return input_ids, modality_inputs


@require_torch
class MossTranscribeDiarizeForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `MossTranscribeDiarizeForConditionalGeneration`.
    """

    model_tester_class = MossTranscribeDiarizeModelTester
    skip_test_audio_features_output_shape = True
    pipeline_model_mapping = (
        {"audio-text-to-text": MossTranscribeDiarizeForConditionalGeneration} if is_torch_available() else {}
    )

    # Override, see qwen3_asr tests for more info.
    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return self.model_tester.encoder_layers

    def _audio_features_get_expected_num_hidden_states(self, model_tester=None):
        return self.model_tester.encoder_layers + 1

    @unittest.skip(
        reason="This test does not apply to MossTranscribeDiarize since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="MossTranscribeDiarize uses input_features_mask and padding_mask instead of audio masks.")
    def test_mismatching_num_audio_tokens(self):
        pass


@require_torch
class MossTranscribeDiarizeForConditionalGenerationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "itazap/MOSS-Transcribe-Diarize-HF"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single_batch_sub_30(self):
        """
        reproducer: https://gist.github.com/itazap/6045ee5b1c4737c5623d5701de68081a
        """
        path = (
            Path(__file__).parent.parent.parent
            / "fixtures/moss_transcribe_diarize/expected_results_single_batch_sub_30.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
        ).to(model.device, dtype=model.dtype)
        torch.testing.assert_close(inputs.input_ids.cpu(), torch.tensor(expected_outputs["input_ids"]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        torch.testing.assert_close(generated_ids.cpu(), torch.tensor(expected_outputs["generated_ids"]))

        decoded_outputs = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(decoded_outputs, expected_outputs["transcriptions"])

    @slow
    def test_single_batch_over_30(self):
        """
        reproducer: https://gist.github.com/itazap/e551c66d2d928be5027c2aa832bc8123
        """
        path = (
            Path(__file__).parent.parent.parent
            / "fixtures/moss_transcribe_diarize/expected_results_single_batch_over_30.json"
        )
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
        ).to(model.device, dtype=model.dtype)
        torch.testing.assert_close(inputs.input_ids.cpu(), torch.tensor(expected_outputs["input_ids"]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        torch.testing.assert_close(generated_ids.cpu(), torch.tensor(expected_outputs["generated_ids"]))

        decoded_outputs = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(decoded_outputs, expected_outputs["transcriptions"])

    @slow
    def test_batched(self):
        """
        reproducer: https://gist.github.com/itazap/549d040019a61b735ae4099da3d7ad1c
        """
        path = Path(__file__).parent.parent.parent / "fixtures/moss_transcribe_diarize/expected_results_batched.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            [
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
            ],
        ).to(model.device, dtype=model.dtype)
        torch.testing.assert_close(inputs.input_ids.cpu(), torch.tensor(expected_outputs["input_ids"]))

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        torch.testing.assert_close(generated_ids.cpu(), torch.tensor(expected_outputs["generated_ids"]))

        decoded_outputs = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(decoded_outputs, expected_outputs["transcriptions"])
