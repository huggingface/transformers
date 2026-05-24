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

import torch

from transformers.testing_utils import require_torch, slow


def get_small_encoder_config():
    """Small encoder config for fast unit tests."""
    from transformers.models.fun_asr_nano.configuration_fun_asr_nano import FunAsrNanoEncoderConfig

    return FunAsrNanoEncoderConfig(
        input_size=560,
        output_size=64,
        attention_heads=4,
        linear_units=128,
        num_blocks=2,
        tp_blocks=1,
        kernel_size=5,
        sanm_shift=0,
        dropout_rate=0.0,
    )


def get_small_model_config():
    """Small full model config for testing."""
    from transformers.models.fun_asr_nano.configuration_fun_asr_nano import (
        FunAsrNanoAdaptorConfig,
        FunAsrNanoConfig,
        FunAsrNanoCtcConfig,
        FunAsrNanoEncoderConfig,
    )

    return FunAsrNanoConfig(
        audio_encoder_config=FunAsrNanoEncoderConfig(
            input_size=560,
            output_size=64,
            attention_heads=4,
            linear_units=128,
            num_blocks=2,
            tp_blocks=1,
            kernel_size=5,
        ),
        adaptor_config=FunAsrNanoAdaptorConfig(
            downsample_rate=1,
            encoder_dim=64,
            llm_dim=64,
            ffn_dim=128,
            num_layers=1,
            attention_heads=4,
        ),
        text_config={
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
        ctc_config=FunAsrNanoCtcConfig(
            vocab_size=100,
            encoder_dim=64,
            decoder_dim=64,
            ffn_dim=128,
            num_layers=1,
            blank_id=99,
        ),
        audio_token_index=999,
    )


@require_torch
class FunAsrNanoEncoderTest(unittest.TestCase):
    def test_forward(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoEncoder

        config = get_small_encoder_config()
        model = FunAsrNanoEncoder(config).eval()
        x = torch.randn(2, 20, 560)
        lens = torch.tensor([20, 15])
        with torch.no_grad():
            out = model(x, lens)
        self.assertEqual(out.last_hidden_state.shape, (2, 20, 64))

    def test_masking(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoEncoder

        config = get_small_encoder_config()
        model = FunAsrNanoEncoder(config).eval()
        x = torch.randn(1, 10, 560)
        with torch.no_grad():
            o1 = model(x, torch.tensor([10]))
            o2 = model(x, torch.tensor([5]))
        self.assertFalse(torch.allclose(o1.last_hidden_state, o2.last_hidden_state))


@require_torch
class FunAsrNanoAdaptorTest(unittest.TestCase):
    def test_forward(self):
        from transformers.models.fun_asr_nano.configuration_fun_asr_nano import FunAsrNanoAdaptorConfig
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoAdaptor

        config = FunAsrNanoAdaptorConfig(
            downsample_rate=1, encoder_dim=64, llm_dim=128, ffn_dim=256, num_layers=1, attention_heads=4
        )
        model = FunAsrNanoAdaptor(config)
        x = torch.randn(2, 20, 64)
        out, olens = model(x, torch.tensor([20, 15]))
        self.assertEqual(out.shape, (2, 20, 128))

    def test_downsampling(self):
        from transformers.models.fun_asr_nano.configuration_fun_asr_nano import FunAsrNanoAdaptorConfig
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoAdaptor

        config = FunAsrNanoAdaptorConfig(
            downsample_rate=2, encoder_dim=64, llm_dim=128, ffn_dim=256, num_layers=0, attention_heads=4
        )
        model = FunAsrNanoAdaptor(config)
        out, olens = model(torch.randn(1, 20, 64), torch.tensor([20]))
        self.assertEqual(out.shape[1], 10)


@require_torch
class FunAsrNanoModelTest(unittest.TestCase):
    def test_init(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration

        config = get_small_model_config()
        model = FunAsrNanoForConditionalGeneration(config)
        self.assertGreater(sum(p.numel() for p in model.parameters()), 0)

    def test_text_only_forward(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration

        config = get_small_model_config()
        model = FunAsrNanoForConditionalGeneration(config).eval()
        input_ids = torch.randint(0, 900, (1, 10))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self.assertEqual(out.logits.shape, (1, 10, 1000))

    def test_audio_forward(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration

        config = get_small_model_config()
        model = FunAsrNanoForConditionalGeneration(config).eval()
        input_ids = torch.cat(
            [torch.randint(0, 900, (1, 5)), torch.full((1, 8), 999), torch.randint(0, 900, (1, 5))], dim=1
        )
        input_features = torch.randn(1, 8, 560)
        feature_lengths = torch.tensor([8])
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                input_features=input_features,
                feature_lengths=feature_lengths,
            )
        self.assertEqual(out.logits.shape[:2], (1, 18))

    def test_loss_computation(self):
        from transformers.models.fun_asr_nano.modeling_fun_asr_nano import FunAsrNanoForConditionalGeneration

        config = get_small_model_config()
        model = FunAsrNanoForConditionalGeneration(config)
        input_ids = torch.randint(0, 900, (1, 10))
        labels = torch.randint(0, 900, (1, 10))
        labels[:, :3] = -100
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
        self.assertIsNotNone(out.loss)
        self.assertTrue(out.loss.requires_grad)


@slow
@require_torch
class FunAsrNanoIntegrationTest(unittest.TestCase):
    """Integration tests with real model (run with RUN_SLOW=1)."""

    def test_real_checkpoint_loading(self):
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


if __name__ == "__main__":
    unittest.main()
