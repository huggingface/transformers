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

import torch

from transformers import (
    AutoConfig,
    GraniteConfig,
    GraniteSpeechNarConfig,
)
from transformers.models.granite_speech_nar.configuration_granite_speech_nar import (
    GraniteSpeechNarEncoderConfig,
    GraniteSpeechNarProjectorConfig,
)
from transformers.models.granite_speech_nar.modeling_granite_speech_nar import (
    GraniteSpeechNarCTCEncoder,
    GraniteSpeechNarForASR,
    GraniteSpeechNarOutput,
    GraniteSpeechNarProjector,
)


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
        bpe_output_dim=52,
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
        assert config.bpe_output_dim is None

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
        assert restored.encoder_config.bpe_output_dim == 52
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

    def test_no_bpe_head(self):
        config = GraniteSpeechNarEncoderConfig(
            num_layers=2,
            hidden_dim=64,
            num_heads=4,
            dim_head=16,
            input_dim=160,
            output_dim=348,
            context_size=50,
            self_conditioning_layer=1,
            bpe_output_dim=None,
        )
        encoder = GraniteSpeechNarCTCEncoder(config).eval()

        features = torch.randn(1, 50, 160)
        out = encoder(features, output_hidden_states=False)

        assert out.logits is None
        assert out.all_hidden_states is None


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


class TestGraniteSpeechNarForASR:
    def test_forward(self):
        config = _make_small_config()
        model = GraniteSpeechNarForASR(config).eval()

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

    def test_transcribe(self):
        config = _make_small_config()
        model = GraniteSpeechNarForASR(config).eval()

        features = torch.randn(1, 60, 160)
        output = model.transcribe(input_features=features)

        assert output.preds is not None
        assert len(output.preds) == 1
        assert isinstance(output.preds[0], torch.Tensor)

    def test_loss(self):
        config = _make_small_config()
        model = GraniteSpeechNarForASR(config).train()

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
        model = GraniteSpeechNarForASR(config).train()

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
        model = GraniteSpeechNarForASR(config).train()

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
        model = GraniteSpeechNarForASR(config).eval()

        features = torch.randn(1, 60, 160)
        with torch.no_grad():
            output = model(input_features=features)

        assert output.loss is None

    def test_output_encoder_logits_flag(self):
        config = _make_small_config()
        model = GraniteSpeechNarForASR(config).eval()

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
        model = GraniteSpeechNarForASR(config).eval()
        granite_model = model.language_model.model

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
        model = GraniteSpeechNarForASR(config)
        for i, layer in enumerate(model.language_model.model.layers):
            assert layer.self_attn.is_causal is False, f"Layer {i} is_causal is not False"
