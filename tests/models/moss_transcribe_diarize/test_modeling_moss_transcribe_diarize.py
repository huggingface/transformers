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

import unittest

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
    def setUp(cls):
        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "itazap/MOSS-Transcribe-Diarize-HF"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single_batch_sub_30(self):
        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        EXPECTED_OUTPUT = [
            "[0.86][S01] Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to[7.81][7.81][S01] minus 20 degrees.[9.53]"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_single_batch_over_30(self):
        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        EXPECTED_OUTPUT = [
            "[0.76][S01] This week, I traveled to Chicago to deliver my final farewell address to the nation,[5.99][5.99][S01] following in the tradition of presidents before me.[8.94][8.94][S01] It was an opportunity to say thank you.[11.74][11.74][S01] Whether we've seen eye to eye, or rarely agreed at all, my conversations with you,[17.14][17.14][S01] the American people, in living rooms and schools, at farms and on factory floors,[23.32][23.32][S01] at diners, and on distant military outposts, all these conversations are what have kept[29.16][29.16][S01] me honest, kept me inspired, and kept me going.[33.15][33.15][S01] Every day, I learned from you.[35.82][35.82][S01] You made me a better president, and you made me a better man.[39.64][39.64][S01] Over the[39.97]"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_batched(self):
        model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype="auto"
        )

        inputs = self.processor.apply_transcription_request(
            [
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/bcn_weather.mp3",
                "https://huggingface.co/datasets/eustlb/audio-samples/resolve/main/obama2.mp3",
            ],
        ).to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.decode(
            outputs[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )

        EXPECTED_OUTPUT = [
            "[0.86][S01] Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to[7.81][7.81][S01] minus 20 degrees.[9.53]",
            "[0.76][S01] This week, I traveled to Chicago to deliver my final farewell address to the nation,[5.99][5.99][S01] following in the tradition of presidents before me.[8.94][8.94][S01] It was an opportunity to say thank you.[11.74][11.74][S01] Whether we've seen eye to eye, or rarely agreed at all, my conversations with you,[17.14][17.14][S01] the American people, in living rooms and schools, at farms and on factory floors,[23.32][23.32][S01] at diners, and on distant military outposts, all these conversations are what have kept[29.16][29.16][S01] me honest, kept me inspired, and kept me going.[33.15][33.15][S01] Every day, I learned from you.[35.82][35.82][S01] You made me a better president, and you made me a better man.[39.64][39.64][S01] Over the[39.97]",
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
