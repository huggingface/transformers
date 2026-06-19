# Copyright 2025 Alibaba DAMO Academy and the HuggingFace Inc. team. All rights reserved.
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
"""Tests for Fun-ASR-Nano model."""

import unittest

from transformers import FunAsrNanoConfig, FunAsrNanoEncoderConfig, Qwen3Config
from transformers.testing_utils import require_torch, require_torch_gpu, slow

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import is_torch_available, torch_device


if is_torch_available():
    import torch

    from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration


class FunAsrNanoModelTester(ALMModelTester):
    config_class = FunAsrNanoConfig
    conditional_generation_class = FunAsrNanoForConditionalGeneration
    text_config_class = Qwen3Config
    audio_config_class = FunAsrNanoEncoderConfig
    audio_config_key = "audio_encoder_config"
    audio_mask_key = "feature_lengths"

    def __init__(self, parent, **kwargs):
        # Fun-ASR-Nano audio features are (batch, time, feature_dim) after LFR.
        kwargs.setdefault("feat_seq_length", 20)
        kwargs.setdefault("num_mel_bins", 80)
        kwargs.setdefault("vocab_size", 100)
        kwargs.setdefault("audio_token_id", 99)
        kwargs.setdefault("audio_token_index", 99)
        kwargs.setdefault("is_training", False)

        # Small Qwen3 text config for common tests.
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 128)

        # Small encoder config.
        kwargs.setdefault("input_size", 80)
        kwargs.setdefault("output_size", 32)
        kwargs.setdefault("attention_heads", 4)
        kwargs.setdefault("linear_units", 64)
        kwargs.setdefault("num_blocks", 2)
        kwargs.setdefault("tp_blocks", 1)
        kwargs.setdefault("kernel_size", 5)
        kwargs.setdefault("sanm_shift", 0)
        kwargs.setdefault("dropout_rate", 0.0)

        # Small auxiliary configs; otherwise defaults create a >1M parameter model.
        kwargs.setdefault(
            "adaptor_config",
            {
                "downsample_rate": 1,
                "encoder_dim": 32,
                "llm_dim": 32,
                "ffn_dim": 64,
                "num_layers": 1,
                "attention_heads": 4,
                "dropout_rate": 0.0,
                "use_low_frame_rate": True,
            },
        )
        kwargs.setdefault(
            "ctc_config",
            {
                "vocab_size": 100,
                "encoder_dim": 32,
                "decoder_dim": 32,
                "ffn_dim": 64,
                "num_layers": 1,
                "downsample_rate": 1,
                "blank_id": 99,
                "dropout_rate": 0.0,
            },
        )

        super().__init__(parent, **kwargs)

    def create_audio_features(self):
        """Fun-ASR-Nano audio features are (batch, time, feature_dim) after LFR."""
        from ...test_modeling_common import floats_tensor

        return floats_tensor([self.batch_size, self.feat_seq_length, self.num_mel_bins])

    def get_audio_embeds_mask(self, audio_mask):
        """Fun-ASR-Nano encoder and adaptor preserve sequence length in the test config."""
        return audio_mask

    def _prepare_modality_inputs(self, input_ids, config):
        input_features = self.create_audio_features()
        feature_lengths = torch.full((self.batch_size,), self.feat_seq_length, dtype=torch.long, device=torch_device)
        positions = torch.arange(self.feat_seq_length, device=torch_device)[None, :]
        audio_mask = positions < feature_lengths[:, None]
        audio_embeds_mask = self.get_audio_embeds_mask(audio_mask)
        num_audio_tokens = audio_embeds_mask.sum(dim=1)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        return input_ids, {"input_features": input_features, "feature_lengths": feature_lengths}


@require_torch
class FunAsrNanoForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """Model tester for `FunAsrNanoForConditionalGeneration`."""

    model_tester_class = FunAsrNanoModelTester
    pipeline_model_mapping = {}

    # The adaptor pools/flattens the audio embeddings, so `get_audio_features().pooler_output` is not the standard
    # `(batch, seq, hidden)` shape the common test expects.
    skip_test_audio_features_output_shape = True

    @unittest.skip(reason="inputs_embeds is the audio-fused path; can't match raw token-only embeddings.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="FunAsrNanoForConditionalGeneration has no separate base model without a generation head.")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(
        reason="The SAN-M encoder uses a custom attention path (encoders0/encoders/tp_encoders) that does not expose "
        "per-layer attentions in the standard way."
    )
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip(
        reason="The SAN-M encoder stacks heterogeneous blocks (encoders0/encoders/tp_encoders), so the per-layer "
        "hidden-state count does not match the single `num_hidden_layers` the common test assumes."
    )
    def test_get_audio_features_hidden_states(self):
        pass


@slow
@require_torch_gpu
class FunAsrNanoIntegrationTest(unittest.TestCase):
    """Integration tests with real checkpoint (run with RUN_SLOW=1).

    Expected outputs from original FunASR:
    - ZH (example/zh.mp3): "开饭时间早上九点至下午五点。"
    - EN (example/en.mp3): "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
    """

    model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)
        cls.processor.tokenizer.padding_side = "left"
        cls.model = FunAsrNanoForConditionalGeneration.from_pretrained(
            cls.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _load_audio(self, filename):
        """Load audio from the model repo."""
        import librosa
        from huggingface_hub import hf_hub_download

        audio_path = hf_hub_download("FunAudioLLM/Fun-ASR-Nano-2512", filename)
        audio, _ = librosa.load(audio_path, sr=self.processor.feature_extractor.sampling_rate)
        return audio

    def _decode_generated(self, generated_ids, inputs):
        """Decode only newly generated tokens, excluding the prompt."""
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _prepare_inputs(self, audio, prompt_text):
        """Prepare model inputs using apply_chat_template."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "audio"},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio, sampling_rate=16000, return_tensors="pt")
        return inputs.to(self.model.device)

    def test_generate_chinese(self):
        """Test Chinese transcription matches expected output."""
        audio = self._load_audio("example/zh.mp3")
        inputs = self._prepare_inputs(audio, "语音转写成中文：")

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        text = self._decode_generated(generated_ids, inputs)[0].strip()

        EXPECTED_TRANSCRIPT = "开饭时间早上九点至下午五点。"
        self.assertEqual(text, EXPECTED_TRANSCRIPT)

    def test_generate_english(self):
        """Test English transcription matches expected output."""
        audio = self._load_audio("example/en.mp3")
        inputs = self._prepare_inputs(audio, "Transcribe the audio:")

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        text = self._decode_generated(generated_ids, inputs)[0].strip()

        EXPECTED_TRANSCRIPT = "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
        self.assertEqual(text, EXPECTED_TRANSCRIPT)

    def test_generate_batch(self):
        """Test batch inference with both Chinese and English audio."""
        audio_zh = self._load_audio("example/zh.mp3")
        audio_en = self._load_audio("example/en.mp3")

        messages_zh = [
            {"role": "user", "content": [{"type": "text", "text": "语音转写成中文："}, {"type": "audio"}]},
        ]
        messages_en = [
            {"role": "user", "content": [{"type": "text", "text": "Transcribe the audio:"}, {"type": "audio"}]},
        ]

        text_zh = self.processor.apply_chat_template(messages_zh, add_generation_prompt=True, tokenize=False)
        text_en = self.processor.apply_chat_template(messages_en, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=[text_zh, text_en],
            audio=[audio_zh, audio_en],
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        transcripts = [text.strip() for text in self._decode_generated(generated_ids, inputs)]

        EXPECTED_TRANSCRIPTS = [
            "开饭时间早上九点至下午五点。",
            "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.",
        ]
        self.assertListEqual(transcripts, EXPECTED_TRANSCRIPTS)


if __name__ == "__main__":
    unittest.main()
