# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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
"""Tests for GraniteSpeechNar model."""

import math
import unittest

import pytest
import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    GraniteConfig,
    GraniteSpeechNarConfig,
)
from transformers.models.granite_speech_nar.configuration_granite_speech_nar import (
    GraniteSpeechNarEncoderConfig,
    GraniteSpeechNarProjectorConfig,
)
from transformers.models.granite_speech_nar.modeling_granite_speech_nar import (
    GraniteSpeechNarCTCEncoder,
    GraniteSpeechNarForCTC,
    GraniteSpeechNarOutput,
    GraniteSpeechNarProjector,
)
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_datasets_available


if is_datasets_available():
    from datasets import load_dataset


def _make_small_config():
    encoder_config = GraniteSpeechNarEncoderConfig(
        num_layers=4,
        hidden_dim=64,
        num_heads=4,
        dim_head=16,
        input_dim=160,
        output_dim=10,
        context_size=50,
        self_conditioning_layer=2,
        bpe_output_dim=51,
        bpe_pooling_window=4,
    )
    projector_config = GraniteSpeechNarProjectorConfig(
        encoder_dim=64,
        llm_dim=128,
        downsample_rate=5,
        num_encoder_layers=4,
        hidden_size=128,
        num_heads=4,
        num_layers=1,
        block_size=15,
    )
    text_config = GraniteConfig(
        vocab_size=51,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        embedding_multiplier=1.0,
        attention_multiplier=1.0,
        residual_multiplier=1.0,
        logits_scaling=1.0,
    )
    return GraniteSpeechNarConfig(
        encoder_config=encoder_config,
        projector_config=projector_config,
        text_config=text_config.to_dict(),
        encoder_layer_indices=[1, 2, 3, -1],
        scale_projected_embeddings=False,
    )


# === Configuration tests ===


class TestConfiguration:
    def test_encoder_config_defaults(self):
        config = GraniteSpeechNarEncoderConfig()
        assert config.model_type == "granite_speech_nar_encoder"
        assert config.input_dim == 160
        assert config.num_layers == 16
        assert config.hidden_dim == 1024
        assert config.self_conditioning_layer == 8
        assert config.bpe_output_dim == 49153

    def test_projector_config_defaults(self):
        config = GraniteSpeechNarProjectorConfig()
        assert config.model_type == "granite_speech_nar_projector"
        assert config.encoder_dim == 1024
        assert config.llm_dim == 2048
        assert config.downsample_rate == 5

    def test_config_defaults(self):
        config = GraniteSpeechNarConfig()
        assert config.model_type == "granite_speech_nar"
        assert config.encoder_layer_indices == [4, 8, 12, -1]
        assert config.scale_projected_embeddings is True

    def test_config_serialization_roundtrip(self):
        config = _make_small_config()
        d = config.to_dict()
        restored = GraniteSpeechNarConfig(**d)
        assert restored.encoder_config.num_layers == 4
        assert restored.encoder_config.bpe_output_dim == 51
        assert restored.projector_config.num_layers == 1
        assert restored.encoder_layer_indices == [1, 2, 3, -1]

    def test_auto_config_resolution(self):
        config = AutoConfig.for_model("granite_speech_nar")
        assert isinstance(config, GraniteSpeechNarConfig)


# === Encoder tests ===


class TestEncoder:
    def test_output_shapes(self):
        config = GraniteSpeechNarEncoderConfig(
            num_layers=4,
            hidden_dim=64,
            num_heads=4,
            dim_head=16,
            input_dim=160,
            output_dim=348,
            context_size=50,
            self_conditioning_layer=2,
            bpe_output_dim=100,
            bpe_pooling_window=4,
        )
        encoder = GraniteSpeechNarCTCEncoder(config).eval()

        B, T = 2, 100
        features = torch.randn(B, T, 160)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, 80:] = False

        out = encoder(features, mask, output_hidden_states=True)

        assert out.logits is not None
        assert out.logits.shape[1] == 100
        assert out.all_hidden_states is not None
        assert len(out.all_hidden_states) == 5  # input + 4 layers


# === Projector tests ===


class TestProjector:
    def test_output_shape(self):
        config = GraniteSpeechNarProjectorConfig(
            encoder_dim=64,
            llm_dim=128,
            downsample_rate=5,
            num_encoder_layers=2,
            hidden_size=128,
            num_heads=4,
            num_layers=1,
            block_size=15,
        )
        projector = GraniteSpeechNarProjector(config)

        B, T = 2, 60
        x = torch.randn(B, T, 2 * 64)
        out = projector(x)
        expected_len = math.ceil(T / config.block_size) * (config.block_size // config.downsample_rate)
        assert out.shape == (B, expected_len, 128)

    def test_handles_non_divisible_length(self):
        config = GraniteSpeechNarProjectorConfig(
            encoder_dim=64,
            llm_dim=128,
            downsample_rate=5,
            num_encoder_layers=1,
            hidden_size=64,
            num_heads=4,
            num_layers=1,
            block_size=15,
        )
        projector = GraniteSpeechNarProjector(config)

        x = torch.randn(1, 37, 64)
        out = projector(x)
        assert out.shape == (1, 9, 128)


# === Full model tests ===


class TestGraniteSpeechNarForCTC:
    def test_forward(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()

        B, T = 2, 100
        features = torch.randn(B, T, 160)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, 80:] = False

        with torch.no_grad():
            output = model(input_features=features, attention_mask=mask)

        assert isinstance(output, GraniteSpeechNarOutput)
        assert output.logits is not None
        assert isinstance(output.logits, list)
        assert len(output.logits) == B
        for logits in output.logits:
            assert logits.ndim == 2
            assert logits.shape[1] == 51

    def test_generate(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()

        features = torch.randn(1, 60, 160)
        output = model.generate(input_features=features)

        assert output.preds is not None
        assert len(output.preds) == 1
        assert isinstance(output.preds[0], torch.Tensor)

    def test_generate_multi_step(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()

        features = torch.randn(2, 80, 160)
        mask = torch.ones(2, 80, dtype=torch.bool)
        mask[1, 60:] = False

        out1 = model.generate(input_features=features, attention_mask=mask, num_editing_steps=1)
        out2 = model.generate(input_features=features, attention_mask=mask, num_editing_steps=3)

        assert out1.preds is not None
        assert out2.preds is not None
        assert len(out1.preds) == 2
        assert len(out2.preds) == 2
        # Multi-step should produce valid predictions (may or may not differ)
        for pred in out2.preds:
            assert isinstance(pred, torch.Tensor)
            assert pred.ndim == 1

    def test_loss(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).train()

        B, T = 2, 100
        features = torch.randn(B, T, 160)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[1, 80:] = False
        labels = torch.randint(0, 51, (B, 5))
        label_lengths = torch.tensor([5, 3])

        output = model(
            input_features=features,
            attention_mask=mask,
            labels=labels,
            label_lengths=label_lengths,
        )

        assert output.loss is not None
        assert output.loss.ndim == 0
        assert output.loss.requires_grad
        output.loss.backward()

    def test_loss_with_ce(self):
        config = _make_small_config()
        config.ce_loss_lambda = 0.5
        model = GraniteSpeechNarForCTC(config).train()

        features = torch.randn(1, 60, 160)
        labels = torch.randint(0, 51, (1, 4))
        label_lengths = torch.tensor([4])

        output = model(
            input_features=features,
            labels=labels,
            label_lengths=label_lengths,
        )

        assert output.loss is not None
        assert output.loss.requires_grad
        output.loss.backward()

    def test_loss_with_encoder_ctc(self):
        config = _make_small_config()
        config.encoder_ctc_loss_lambda = 0.3
        model = GraniteSpeechNarForCTC(config).train()

        features = torch.randn(1, 60, 160)
        labels = torch.randint(0, 51, (1, 4))
        label_lengths = torch.tensor([4])

        output = model(
            input_features=features,
            labels=labels,
            label_lengths=label_lengths,
        )

        assert output.loss is not None
        assert output.loss.requires_grad
        output.loss.backward()

    def test_no_loss_without_labels(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()

        features = torch.randn(1, 60, 160)
        with torch.no_grad():
            output = model(input_features=features)

        assert output.loss is None

    def test_output_encoder_logits_flag(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()

        features = torch.randn(1, 60, 160)
        with torch.no_grad():
            out_no = model(input_features=features, output_encoder_logits=False)
            out_yes = model(input_features=features, output_encoder_logits=True)

        assert out_no.encoder_logits is None
        assert out_yes.encoder_logits is not None
        assert out_no.encoder_preds is not None  # always returned

    def test_automodel_resolves(self):
        config = AutoConfig.for_model("granite_speech_nar")
        assert isinstance(config, GraniteSpeechNarConfig)
        assert config.model_type == "granite_speech_nar"


# === Bidirectional attention test ===


class TestBidirectionalAttention:
    def test_last_token_affects_first(self):
        """Changing the last token must affect the first (bidirectional)."""
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config).eval()
        granite_model = model.model.language_model

        embeds_a = torch.randn(1, 10, 128)
        embeds_b = embeds_a.clone()
        embeds_b[0, -1, :] = torch.randn(128)

        with torch.no_grad():
            out_a = granite_model(inputs_embeds=embeds_a).last_hidden_state
            out_b = granite_model(inputs_embeds=embeds_b).last_hidden_state

        diff_first = (out_a[0, 0] - out_b[0, 0]).abs().max().item()
        assert diff_first > 1e-5, f"First token unchanged (diff={diff_first}). Attention appears causal."

    def test_is_causal_false_on_layers(self):
        config = _make_small_config()
        model = GraniteSpeechNarForCTC(config)
        for i, layer in enumerate(model.model.language_model.layers):
            assert layer.self_attn.is_causal is False, f"Layer {i} is_causal is not False"


# === Integration tests ===


@require_torch
class GraniteSpeechNarIntegrationTest(unittest.TestCase):
    model_path = "ibm-granite/granite-speech-4.1-2b-nar"
    _dataset = None

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        samples = self._dataset.sort("id")[:num_samples]["audio"]
        return [torch.tensor(x["array"], dtype=torch.float32) for x in samples]

    @slow
    def test_single_sample_transcription(self):
        model = AutoModel.from_pretrained(
            self.model_path, attn_implementation="flash_attention_2", device_map=torch_device, dtype=torch.bfloat16
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_path)

        waveforms = self._load_datasamples(1)
        inputs = processor(waveforms, device=torch_device)
        output = model.generate(**inputs)
        transcriptions = processor.batch_decode(output.preds)

        expected = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"
        self.assertEqual(transcriptions[0], expected)

    @slow
    def test_batch_transcription(self):
        model = AutoModel.from_pretrained(
            self.model_path, attn_implementation="flash_attention_2", device_map=torch_device, dtype=torch.bfloat16
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_path)

        waveforms = self._load_datasamples(2)
        inputs = processor(waveforms, device=torch_device)
        output = model.generate(**inputs)
        transcriptions = processor.batch_decode(output.preds)

        expected = [
            "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "nor is mister quilter's manner less interesting than his matter",
        ]
        self.assertEqual(len(transcriptions), 2)
        self.assertEqual(transcriptions, expected)

    @slow
    @pytest.mark.skipif(not is_datasets_available(), reason="datasets not installed")
    def test_processor_output_shapes(self):
        processor = AutoProcessor.from_pretrained(self.model_path)

        waveforms = self._load_datasamples(2)
        inputs = processor(waveforms, device="cpu")

        self.assertEqual(inputs["input_features"].ndim, 3)
        self.assertEqual(inputs["input_features"].shape[0], 2)
        self.assertEqual(inputs["input_features"].shape[2], 160)

        self.assertEqual(inputs["attention_mask"].shape, inputs["input_features"].shape[:2])

        # Shorter sample should have False values at end
        mask_sums = inputs["attention_mask"].sum(dim=1)
        self.assertEqual(mask_sums[0].item(), inputs["input_features"].shape[1])
        self.assertLess(mask_sums[1].item(), mask_sums[0].item())
