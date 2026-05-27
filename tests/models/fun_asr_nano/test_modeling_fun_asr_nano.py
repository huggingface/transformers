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

from transformers import FunAsrNanoConfig, FunAsrNanoEncoderConfig
from transformers.testing_utils import require_torch, slow

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import is_torch_available


if is_torch_available():
    import torch

    from transformers import FunAsrNanoForConditionalGeneration


class FunAsrNanoModelTester(ALMModelTester):
    config_class = FunAsrNanoConfig
    conditional_generation_class = FunAsrNanoForConditionalGeneration
    text_config_class = None  # will use auto
    audio_config_class = FunAsrNanoEncoderConfig
    audio_config_key = "audio_encoder_config"
    audio_mask_key = None  # Fun-ASR-Nano uses feature_lengths, not a mask tensor

    def __init__(self, parent, **kwargs):
        # Fun-ASR-Nano specific: audio features are (batch, time, 560) not (batch, mel, time)
        kwargs.setdefault("feat_seq_length", 20)
        kwargs.setdefault("num_mel_bins", 560)
        kwargs.setdefault("audio_token_id", 999)

        # Small encoder config
        kwargs.setdefault(
            "audio_config",
            {
                "model_type": "fun_asr_nano_encoder",
                "input_size": 560,
                "output_size": 64,
                "attention_heads": 4,
                "linear_units": 128,
                "num_blocks": 2,
                "tp_blocks": 1,
                "kernel_size": 5,
                "sanm_shift": 0,
                "dropout_rate": 0.0,
            },
        )

        # Small text config
        kwargs.setdefault(
            "text_config",
            {
                "model_type": "qwen3",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 1000,
                "max_position_embeddings": 512,
                "head_dim": 16,
            },
        )

        super().__init__(parent, **kwargs)

    def create_audio_features(self):
        """Fun-ASR-Nano audio features are (batch, time, feature_dim) after LFR."""
        from ...test_modeling_common import floats_tensor

        return floats_tensor([self.batch_size, self.feat_seq_length, self.num_mel_bins])

    def get_audio_embeds_mask(self, audio_mask):
        """Fun-ASR-Nano encoder preserves sequence length (no downsampling in encoder)."""
        # The adaptor with downsample_rate=1 also preserves length
        return audio_mask


@require_torch
class FunAsrNanoForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """Model tester for `FunAsrNanoForConditionalGeneration`."""

    model_tester_class = FunAsrNanoModelTester
    pipeline_model_mapping = {}

    @unittest.skip(reason="inputs_embeds is the audio-fused path; can't match raw token-only embeddings.")
    def test_inputs_embeds_matches_input_ids(self):
        pass


@slow
@require_torch
class FunAsrNanoIntegrationTest(unittest.TestCase):
    """Integration tests with real checkpoint (run with RUN_SLOW=1).

    Expected outputs from original FunASR:
    - ZH (example/zh.mp3): "开饭时间早上九点至下午五点。"
    - EN (example/en.mp3): "The tribal chieftain called for the boy, and presented him with fifty pieces of gold."
    """

    model_id = "FunAudioLLM/Fun-ASR-Nano-2512-hf"

    def test_checkpoint_weight_counts(self):
        """Verify all weights from the original checkpoint load correctly."""
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download("FunAudioLLM/Fun-ASR-Nano-2512", "model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

        enc_keys = sum(1 for k in ckpt if k.startswith("audio_encoder."))
        adp_keys = sum(1 for k in ckpt if k.startswith("audio_adaptor."))
        llm_keys = sum(1 for k in ckpt if k.startswith("llm."))

        self.assertEqual(enc_keys, 914)
        self.assertEqual(adp_keys, 36)
        self.assertEqual(llm_keys, 311)
        self.assertEqual(enc_keys + adp_keys + llm_keys, len(ckpt))

    def test_model_load_and_param_count(self):
        """Verify model loads and has expected parameter count (~830M)."""
        model = FunAsrNanoForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 800_000_000)
        self.assertLess(total_params, 900_000_000)


if __name__ == "__main__":
    unittest.main()
