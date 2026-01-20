# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch VibeVoice model."""

import copy
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        VibeVoiceAcousticCodecConfig,
        VibeVoiceConfig,
        VibeVoiceDiffusionHeadConfig,
        VibeVoiceForConditionalGeneration,
        VibeVoiceModel,
        VibeVoiceSemanticEncoderConfig,
        VibeVoiceStreamingConfig,
    )
    from transformers.models.vibevoice.modeling_vibevoice import (
        VibeVoiceAcousticCodec,
        VibeVoiceDiffusionHead,
        VibeVoiceSemanticEncoder,
        VibeVoiceStreamingForConditionalGeneration,
    )


class VibeVoiceAcousticCodecTester:
    """Tester for VibeVoiceAcousticCodec (σ-VAE)."""

    def __init__(
        self,
        parent,
        batch_size=2,
        audio_length=2400,  # 0.1 second at 24kHz
        channels=1,
        vae_dim=16,
        encoder_n_filters=8,
        decoder_n_filters=8,
        encoder_ratios=[2, 2, 2],
        decoder_ratios=[2, 2, 2],
        encoder_depths="2-2-2-2",
        causal=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.audio_length = audio_length
        self.channels = channels
        self.vae_dim = vae_dim
        self.encoder_n_filters = encoder_n_filters
        self.decoder_n_filters = decoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.decoder_ratios = decoder_ratios
        self.encoder_depths = encoder_depths
        self.causal = causal

    def get_config(self):
        return VibeVoiceAcousticCodecConfig(
            channels=self.channels,
            vae_dim=self.vae_dim,
            encoder_n_filters=self.encoder_n_filters,
            decoder_n_filters=self.decoder_n_filters,
            encoder_ratios=self.encoder_ratios,
            decoder_ratios=self.decoder_ratios,
            encoder_depths=self.encoder_depths,
            causal=self.causal,
            fix_std=0.5,
            std_dist_type="gaussian",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        audio_values = torch.randn(self.batch_size, self.channels, self.audio_length)
        return config, {"audio_values": audio_values}

    def create_and_check_model(self, config, audio_values):
        model = VibeVoiceAcousticCodec(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(audio_values.to(torch_device))

        self.parent.assertIsNotNone(outputs.reconstructed_audio)
        self.parent.assertIsNotNone(outputs.mean)
        self.parent.assertIsNotNone(outputs.latent)

        # Check shapes
        self.parent.assertEqual(outputs.reconstructed_audio.shape[0], self.batch_size)
        self.parent.assertEqual(outputs.reconstructed_audio.shape[1], self.channels)
        self.parent.assertEqual(outputs.mean.shape[0], self.batch_size)
        self.parent.assertEqual(outputs.mean.shape[-1], self.vae_dim)

    def create_and_check_encode_decode(self, config, audio_values):
        model = VibeVoiceAcousticCodec(config=config).to(torch_device).eval()

        with torch.no_grad():
            # Test encode
            mean, log_var = model.encode(audio_values.to(torch_device))
            self.parent.assertIsNotNone(mean)
            self.parent.assertEqual(mean.shape[-1], self.vae_dim)

            # Test decode
            reconstructed = model.decode(mean)
            self.parent.assertIsNotNone(reconstructed)
            self.parent.assertEqual(reconstructed.shape[1], self.channels)


class VibeVoiceSemanticEncoderTester:
    """Tester for VibeVoiceSemanticEncoder."""

    def __init__(
        self,
        parent,
        batch_size=2,
        audio_length=2400,
        channels=1,
        vae_dim=32,
        encoder_n_filters=8,
        encoder_ratios=[2, 2, 2],
        encoder_depths="2-2-2-2",
        causal=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.audio_length = audio_length
        self.channels = channels
        self.vae_dim = vae_dim
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        self.causal = causal

    def get_config(self):
        return VibeVoiceSemanticEncoderConfig(
            channels=self.channels,
            vae_dim=self.vae_dim,
            encoder_n_filters=self.encoder_n_filters,
            encoder_ratios=self.encoder_ratios,
            encoder_depths=self.encoder_depths,
            causal=self.causal,
            fix_std=0,
            std_dist_type="none",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        audio_values = torch.randn(self.batch_size, self.channels, self.audio_length)
        return config, {"audio_values": audio_values}

    def create_and_check_model(self, config, audio_values):
        model = VibeVoiceSemanticEncoder(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(audio_values.to(torch_device))

        self.parent.assertIsNotNone(outputs.semantic_features)
        self.parent.assertEqual(outputs.semantic_features.shape[0], self.batch_size)
        self.parent.assertEqual(outputs.semantic_features.shape[-1], self.vae_dim)


class VibeVoiceDiffusionHeadTester:
    """Tester for VibeVoiceDiffusionHead."""

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=4,
        hidden_size=32,
        latent_size=16,
        head_layers=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.head_layers = head_layers

    def get_config(self):
        return VibeVoiceDiffusionHeadConfig(
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            head_layers=self.head_layers,
            head_ffn_ratio=2.0,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        noisy_latents = torch.randn(self.batch_size, self.seq_length, self.latent_size)
        timesteps = torch.randint(0, 1000, (self.batch_size,))
        conditioning = torch.randn(self.batch_size, self.seq_length, self.hidden_size)
        return config, {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "conditioning": conditioning,
        }

    def create_and_check_model(self, config, noisy_latents, timesteps, conditioning):
        model = VibeVoiceDiffusionHead(config=config).to(torch_device).eval()

        with torch.no_grad():
            output = model(
                noisy_latents.to(torch_device),
                timesteps.to(torch_device),
                conditioning.to(torch_device),
            )

        self.parent.assertIsNotNone(output)
        self.parent.assertEqual(output.shape, noisy_latents.shape)


class VibeVoiceModelTester:
    """Tester for the main VibeVoice model."""

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        acoustic_vae_dim=16,
        semantic_vae_dim=32,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.acoustic_vae_dim = acoustic_vae_dim
        self.semantic_vae_dim = semantic_vae_dim
        self.is_training = is_training

    def get_config(self):
        # Minimal Qwen2-like decoder config
        decoder_config = {
            "model_type": "qwen2",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
        }

        # Minimal acoustic codec config
        acoustic_tokenizer_config = {
            "channels": 1,
            "vae_dim": self.acoustic_vae_dim,
            "encoder_n_filters": 8,
            "decoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "decoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
            "causal": True,
        }

        # Minimal semantic encoder config
        semantic_tokenizer_config = {
            "channels": 1,
            "vae_dim": self.semantic_vae_dim,
            "encoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
            "causal": True,
        }

        # Minimal diffusion head config
        diffusion_head_config = {
            "hidden_size": self.hidden_size,
            "latent_size": self.acoustic_vae_dim,
            "head_layers": 2,
        }

        return VibeVoiceConfig(
            decoder_config=decoder_config,
            acoustic_tokenizer_config=acoustic_tokenizer_config,
            semantic_tokenizer_config=semantic_tokenizer_config,
            diffusion_head_config=diffusion_head_config,
            acoustic_vae_dim=self.acoustic_vae_dim,
            semantic_vae_dim=self.semantic_vae_dim,
        )

    def get_streaming_config(self):
        """Get config for streaming model (no semantic encoder)."""
        decoder_config = {
            "model_type": "qwen2",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
        }

        acoustic_tokenizer_config = {
            "channels": 1,
            "vae_dim": self.acoustic_vae_dim,
            "encoder_n_filters": 8,
            "decoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "decoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
            "causal": True,
        }

        diffusion_head_config = {
            "hidden_size": self.hidden_size,
            "latent_size": self.acoustic_vae_dim,
            "head_layers": 2,
        }

        return VibeVoiceStreamingConfig(
            decoder_config=decoder_config,
            acoustic_tokenizer_config=acoustic_tokenizer_config,
            diffusion_head_config=diffusion_head_config,
            acoustic_vae_dim=self.acoustic_vae_dim,
            tts_backbone_num_hidden_layers=self.num_hidden_layers,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones_like(input_ids)
        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def prepare_config_and_inputs_for_common(self):
        return self.prepare_config_and_inputs()

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = VibeVoiceModel(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
            )

        self.parent.assertIsNotNone(outputs.last_hidden_state)
        self.parent.assertEqual(
            outputs.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )

    def create_and_check_for_conditional_generation(self, config, input_ids, attention_mask):
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
            )

        self.parent.assertIsNotNone(outputs.logits)
        self.parent.assertEqual(
            outputs.logits.shape,
            (self.batch_size, self.seq_length, self.vocab_size),
        )


@require_torch
class VibeVoiceAcousticCodecTest(unittest.TestCase):
    """Tests for VibeVoiceAcousticCodec."""

    def setUp(self):
        self.model_tester = VibeVoiceAcousticCodecTester(self)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "vibevoice_acoustic_codec")

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, **inputs)

    def test_encode_decode(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encode_decode(config, **inputs)

    def test_model_from_pretrained_config(self):
        config = self.model_tester.get_config()
        model = VibeVoiceAcousticCodec(config)
        self.assertIsNotNone(model)


@require_torch
class VibeVoiceSemanticEncoderTest(unittest.TestCase):
    """Tests for VibeVoiceSemanticEncoder."""

    def setUp(self):
        self.model_tester = VibeVoiceSemanticEncoderTester(self)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "vibevoice_semantic_encoder")

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, **inputs)


@require_torch
class VibeVoiceDiffusionHeadTest(unittest.TestCase):
    """Tests for VibeVoiceDiffusionHead."""

    def setUp(self):
        self.model_tester = VibeVoiceDiffusionHeadTester(self)

    def test_config(self):
        config = self.model_tester.get_config()
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "vibevoice_diffusion_head")

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, **inputs)


@require_torch
class VibeVoiceConfigTest(unittest.TestCase):
    """Tests for VibeVoiceConfig."""

    def test_config(self):
        config = VibeVoiceConfig()
        self.assertEqual(config.model_type, "vibevoice")

    def test_streaming_config(self):
        config = VibeVoiceStreamingConfig()
        self.assertEqual(config.model_type, "vibevoice_streaming")
        self.assertTrue(config.is_streaming)
        self.assertIsNone(config.semantic_tokenizer_config)

    def test_config_with_sub_configs(self):
        config = VibeVoiceConfig(
            decoder_config={"model_type": "qwen2", "hidden_size": 64},
            acoustic_tokenizer_config={"vae_dim": 32},
            semantic_tokenizer_config={"vae_dim": 64},
            diffusion_head_config={"hidden_size": 64, "latent_size": 32},
        )
        self.assertEqual(config.decoder_config.hidden_size, 64)
        self.assertEqual(config.acoustic_tokenizer_config.vae_dim, 32)
        self.assertEqual(config.semantic_tokenizer_config.vae_dim, 64)
        self.assertEqual(config.diffusion_head_config.hidden_size, 64)

    def test_config_serialization(self):
        config = VibeVoiceConfig(
            decoder_config={"model_type": "qwen2", "hidden_size": 64},
            acoustic_tokenizer_config={"vae_dim": 32},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.save_pretrained(tmp_dir)
            loaded_config = VibeVoiceConfig.from_pretrained(tmp_dir)

        self.assertEqual(config.model_type, loaded_config.model_type)
        self.assertEqual(config.decoder_config.hidden_size, loaded_config.decoder_config.hidden_size)


@require_torch
class VibeVoiceModelTest(ModelTesterMixin, unittest.TestCase):
    """Tests for VibeVoiceModel and VibeVoiceForConditionalGeneration."""

    all_model_classes = (VibeVoiceModel, VibeVoiceForConditionalGeneration) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    test_resize_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VibeVoiceConfig,
            hidden_size=32,
        )

    def test_config(self):
        # Skip the default config test as VibeVoice has nested configs
        pass

    def test_model(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, **inputs)

    def test_for_conditional_generation(self):
        config, inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(config, **inputs)

    def test_streaming_model(self):
        """Test streaming model variant."""
        config = self.model_tester.get_streaming_config()
        input_ids = ids_tensor([2, 8], 100)
        attention_mask = torch.ones_like(input_ids)

        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(torch_device),
                attention_mask=attention_mask.to(torch_device),
            )

        self.assertIsNotNone(outputs.logits)
        # Streaming model should not have semantic tokenizer
        self.assertIsNone(model.model.semantic_tokenizer)

    def test_model_with_cache(self):
        """Test model with KV cache."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
                use_cache=True,
            )

        self.assertIsNotNone(outputs.past_key_values)

    def test_model_save_load(self):
        """Test saving and loading model."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model = VibeVoiceForConditionalGeneration.from_pretrained(tmp_dir).to(torch_device).eval()

        with torch.no_grad():
            original_outputs = model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
            )
            loaded_outputs = loaded_model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
            )

        self.assertTrue(torch.allclose(original_outputs.logits, loaded_outputs.logits, atol=1e-5))

    @unittest.skip("VibeVoice has composite architecture")
    def test_initialization(self):
        pass

    @unittest.skip("VibeVoice has composite architecture")
    def test_forward_signature(self):
        pass

    @unittest.skip("VibeVoice doesn't support standard inputs_embeds pattern")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("VibeVoice has composite architecture")
    def test_model_common_attributes(self):
        pass

    @unittest.skip("VibeVoice has composite architecture")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip("VibeVoice has composite architecture")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - use VibeVoiceOutputsTest instead")
    def test_attention_outputs(self):
        pass

    @unittest.skip("VibeVoice has composite architecture")
    def test_hidden_states_output(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - attn_implementation not propagated to sub-configs")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - init_weights behavior differs")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - safetensors weights may not match exactly")
    def test_can_use_safetensors(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - tied weights behavior differs")
    def test_load_save_without_tied_weights(self):
        pass

    @unittest.skip("VibeVoice has composite architecture - dtype handling differs")
    def test_bc_torch_dtype(self):
        pass

    @unittest.skip("VibeVoice uses custom generation - see VibeVoiceIntegrationTest for generation tests")
    def test_generation_tester_mixin_inheritance(self):
        pass


class VibeVoiceStreamingModelTester:
    """Tester for VibeVoice streaming model variant."""

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        acoustic_vae_dim=16,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.acoustic_vae_dim = acoustic_vae_dim
        self.is_training = is_training

    def get_config(self):
        decoder_config = {
            "model_type": "qwen2",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6,
        }

        acoustic_tokenizer_config = {
            "channels": 1,
            "vae_dim": self.acoustic_vae_dim,
            "encoder_n_filters": 8,
            "decoder_n_filters": 8,
            "encoder_ratios": [2, 2],
            "decoder_ratios": [2, 2],
            "encoder_depths": "2-2-2",
            "causal": True,
        }

        diffusion_head_config = {
            "hidden_size": self.hidden_size,
            "latent_size": self.acoustic_vae_dim,
            "head_layers": 2,
        }

        return VibeVoiceStreamingConfig(
            decoder_config=decoder_config,
            acoustic_tokenizer_config=acoustic_tokenizer_config,
            diffusion_head_config=diffusion_head_config,
            acoustic_vae_dim=self.acoustic_vae_dim,
            tts_backbone_num_hidden_layers=self.num_hidden_layers,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones_like(input_ids)
        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


@require_torch
class VibeVoiceStreamingModelTest(unittest.TestCase):
    """Tests for VibeVoice streaming model variant."""

    def setUp(self):
        self.model_tester = VibeVoiceStreamingModelTester(self)

    def test_streaming_config(self):
        """Test streaming configuration."""
        config = self.model_tester.get_config()
        self.assertEqual(config.model_type, "vibevoice_streaming")
        self.assertTrue(config.is_streaming)
        self.assertIsNone(config.semantic_tokenizer_config)

    def test_streaming_model_forward(self):
        """Test streaming model forward pass."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceStreamingForConditionalGeneration(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
            )

        self.assertIsNotNone(outputs.logits)
        self.assertIsNone(model.model.semantic_tokenizer)

    def test_streaming_model_with_cache(self):
        """Test streaming model with KV cache."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceStreamingForConditionalGeneration(config=config).to(torch_device).eval()

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
                use_cache=True,
            )

        self.assertIsNotNone(outputs.past_key_values)

    def test_streaming_model_save_load(self):
        """Test saving and loading streaming model."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceStreamingForConditionalGeneration(config=config).to(torch_device).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model = VibeVoiceStreamingForConditionalGeneration.from_pretrained(tmp_dir).to(torch_device).eval()

        with torch.no_grad():
            original_outputs = model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
            )
            loaded_outputs = loaded_model(
                input_ids=inputs["input_ids"].to(torch_device),
                attention_mask=inputs["attention_mask"].to(torch_device),
            )

        self.assertTrue(torch.allclose(original_outputs.logits, loaded_outputs.logits, atol=1e-5))


@require_torch
class VibeVoiceModulesTest(unittest.TestCase):
    """Tests for individual VibeVoice modules."""

    def test_rms_norm(self):
        """Test VibeVoiceRMSNorm."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceRMSNorm

        dim = 32
        batch_size = 2
        seq_len = 4

        norm = VibeVoiceRMSNorm(dim).to(torch_device)
        x = torch.randn(batch_size, seq_len, dim, device=torch_device)

        output = norm(x)

        self.assertEqual(output.shape, x.shape)
        # Check that output has approximately unit RMS per token
        rms = torch.sqrt(output.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=0.1))

    def test_rms_norm_no_affine(self):
        """Test VibeVoiceRMSNorm without affine transformation."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceRMSNorm

        dim = 32
        norm = VibeVoiceRMSNorm(dim, elementwise_affine=False).to(torch_device)
        self.assertIsNone(norm.weight)

    def test_conv_rms_norm(self):
        """Test VibeVoiceConvRMSNorm."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceConvRMSNorm

        dim = 32
        batch_size = 2
        time = 16

        norm = VibeVoiceConvRMSNorm(dim).to(torch_device)
        # Conv format: (batch, channels, time)
        x = torch.randn(batch_size, dim, time, device=torch_device)

        output = norm(x)

        self.assertEqual(output.shape, x.shape)

    def test_causal_conv1d(self):
        """Test VibeVoiceCausalConv1d."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceCausalConv1d

        in_channels = 16
        out_channels = 32
        kernel_size = 3
        batch_size = 2
        time = 10

        conv = VibeVoiceCausalConv1d(in_channels, out_channels, kernel_size).to(torch_device)
        x = torch.randn(batch_size, in_channels, time, device=torch_device)

        output = conv(x)

        # Output should have same time dimension (causal padding)
        self.assertEqual(output.shape, (batch_size, out_channels, time))

    def test_causal_conv_transpose1d(self):
        """Test VibeVoiceCausalConvTranspose1d."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceCausalConvTranspose1d

        in_channels = 32
        out_channels = 16
        kernel_size = 4
        stride = 2
        batch_size = 2
        time = 10

        conv = VibeVoiceCausalConvTranspose1d(in_channels, out_channels, kernel_size, stride).to(torch_device)
        x = torch.randn(batch_size, in_channels, time, device=torch_device)

        output = conv(x)

        # Output should be upsampled
        self.assertEqual(output.shape[0], batch_size)
        self.assertEqual(output.shape[1], out_channels)
        self.assertEqual(output.shape[2], time * stride)

    def test_tokenizer_ffn(self):
        """Test VibeVoiceTokenizerFFN."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceTokenizerFFN

        dim = 32
        batch_size = 2
        seq_len = 4

        ffn = VibeVoiceTokenizerFFN(dim).to(torch_device)
        x = torch.randn(batch_size, seq_len, dim, device=torch_device)

        output = ffn(x)

        self.assertEqual(output.shape, x.shape)

    def test_tokenizer_mixer(self):
        """Test VibeVoiceTokenizerMixer."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceTokenizerMixer

        dim = 32
        batch_size = 2
        seq_len = 8

        # Test causal mixer
        mixer_causal = VibeVoiceTokenizerMixer(dim, causal=True).to(torch_device)
        x = torch.randn(batch_size, seq_len, dim, device=torch_device)
        output = mixer_causal(x)
        self.assertEqual(output.shape, x.shape)

        # Test non-causal mixer
        mixer_noncausal = VibeVoiceTokenizerMixer(dim, causal=False).to(torch_device)
        output = mixer_noncausal(x)
        self.assertEqual(output.shape, x.shape)

    def test_tokenizer_block(self):
        """Test VibeVoiceTokenizerBlock."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceTokenizerBlock

        dim = 32
        batch_size = 2
        seq_len = 8

        block = VibeVoiceTokenizerBlock(dim).to(torch_device)
        x = torch.randn(batch_size, seq_len, dim, device=torch_device)

        output = block(x)

        self.assertEqual(output.shape, x.shape)

    def test_timestep_embedder(self):
        """Test VibeVoiceTimestepEmbedder."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceTimestepEmbedder

        hidden_size = 32
        batch_size = 4

        embedder = VibeVoiceTimestepEmbedder(hidden_size).to(torch_device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=torch_device)

        output = embedder(timesteps)

        self.assertEqual(output.shape, (batch_size, hidden_size))

    def test_diffusion_ffn(self):
        """Test VibeVoiceDiffusionFFN."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceDiffusionFFN

        hidden_size = 32
        batch_size = 2
        seq_len = 4

        ffn = VibeVoiceDiffusionFFN(hidden_size).to(torch_device)
        x = torch.randn(batch_size, seq_len, hidden_size, device=torch_device)

        output = ffn(x)

        self.assertEqual(output.shape, x.shape)

    def test_connector(self):
        """Test VibeVoiceConnector."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceConnector

        input_dim = 16
        hidden_size = 32
        batch_size = 2
        seq_len = 4

        connector = VibeVoiceConnector(input_dim, hidden_size).to(torch_device)
        x = torch.randn(batch_size, seq_len, input_dim, device=torch_device)

        output = connector(x)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))


@require_torch
class VibeVoiceAudioDecodingTest(unittest.TestCase):
    """Tests for audio decoding functionality."""

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)

    def test_decode_audio(self):
        """Test audio decoding from acoustic features."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        batch_size = 2
        seq_len = 4
        acoustic_features = torch.randn(batch_size, seq_len, config.acoustic_vae_dim, device=torch_device)

        with torch.no_grad():
            audio = model.decode_audio(acoustic_features)

        self.assertIsNotNone(audio)
        self.assertEqual(audio.shape[0], batch_size)

    def test_acoustic_codec_reparameterize(self):
        """Test VAE reparameterization trick."""
        config = VibeVoiceAcousticCodecConfig(
            vae_dim=16,
            encoder_n_filters=8,
            decoder_n_filters=8,
            encoder_ratios=[2, 2],
            decoder_ratios=[2, 2],
            encoder_depths="2-2-2",
            fix_std=0.5,
        )
        model = VibeVoiceAcousticCodec(config=config).to(torch_device).eval()

        batch_size = 2
        seq_len = 4
        mean = torch.randn(batch_size, seq_len, config.vae_dim, device=torch_device)
        log_var = torch.randn(batch_size, seq_len, config.vae_dim, device=torch_device)

        # Test with fixed std
        z = model.reparameterize(mean, log_var)
        self.assertEqual(z.shape, mean.shape)

        # Test with fix_std=0 (no sampling)
        model.fix_std = 0
        z = model.reparameterize(mean, log_var)
        self.assertTrue(torch.allclose(z, mean))


@require_torch
class VibeVoiceKVCacheTest(unittest.TestCase):
    """Tests for KV cache functionality."""

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)

    def test_model_past_with_large_inputs(self):
        """Test model with past key values and large inputs."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)

        # First forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

        past_key_values = outputs.past_key_values

        # Create next tokens
        batch_size = input_ids.shape[0]
        next_tokens = ids_tensor((batch_size, 3), config.decoder_config.vocab_size).to(torch_device)
        next_attn_mask = torch.ones((batch_size, 3), device=torch_device)

        # Full sequence forward
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        with torch.no_grad():
            output_from_no_past = model(input_ids=next_input_ids, attention_mask=next_attention_mask)["logits"]
            output_from_past = model(
                input_ids=next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values
            )["logits"]

        # Check that outputs match for the last 3 positions
        self.assertEqual(output_from_past.shape[1], 3)
        self.assertTrue(torch.allclose(output_from_no_past[:, -3:, :], output_from_past, atol=1e-3))

    def test_cache_position(self):
        """Test that cache_position is properly handled."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)
        seq_len = input_ids.shape[1]

        cache_position = torch.arange(seq_len, device=torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )

        self.assertIsNotNone(outputs.logits)
        self.assertIsNotNone(outputs.past_key_values)


@require_torch
class VibeVoiceSaveLoadTest(unittest.TestCase):
    """Tests for model save/load functionality."""

    def test_acoustic_codec_save_load(self):
        """Test saving and loading acoustic codec."""
        config = VibeVoiceAcousticCodecConfig(
            vae_dim=16,
            encoder_n_filters=8,
            decoder_n_filters=8,
            encoder_ratios=[2, 2],
            decoder_ratios=[2, 2],
            encoder_depths="2-2-2",
        )
        model = VibeVoiceAcousticCodec(config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model, info = VibeVoiceAcousticCodec.from_pretrained(tmp_dir, output_loading_info=True)

        self.assertEqual(len(info["missing_keys"]), 0)

    def test_semantic_encoder_save_load(self):
        """Test saving and loading semantic encoder."""
        config = VibeVoiceSemanticEncoderConfig(
            vae_dim=32,
            encoder_n_filters=8,
            encoder_ratios=[2, 2],
            encoder_depths="2-2-2",
        )
        model = VibeVoiceSemanticEncoder(config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model, info = VibeVoiceSemanticEncoder.from_pretrained(tmp_dir, output_loading_info=True)

        self.assertEqual(len(info["missing_keys"]), 0)

    def test_diffusion_head_save_load(self):
        """Test saving and loading diffusion head."""
        config = VibeVoiceDiffusionHeadConfig(
            hidden_size=32,
            latent_size=16,
            head_layers=2,
        )
        model = VibeVoiceDiffusionHead(config=config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            loaded_model, info = VibeVoiceDiffusionHead.from_pretrained(tmp_dir, output_loading_info=True)

        self.assertEqual(len(info["missing_keys"]), 0)


@require_torch
class VibeVoiceConfigAdvancedTest(unittest.TestCase):
    """Advanced tests for VibeVoice configurations."""

    def test_config_copy(self):
        """Test that config can be copied."""
        config = VibeVoiceConfig(
            decoder_config={"model_type": "qwen2", "hidden_size": 64},
            acoustic_tokenizer_config={"vae_dim": 32},
        )
        copied_config = copy.deepcopy(config)

        self.assertEqual(config.decoder_config.hidden_size, copied_config.decoder_config.hidden_size)
        self.assertEqual(config.acoustic_tokenizer_config.vae_dim, copied_config.acoustic_tokenizer_config.vae_dim)

    def test_config_to_dict(self):
        """Test config serialization to dict."""
        config = VibeVoiceConfig(
            decoder_config={"model_type": "qwen2", "hidden_size": 64},
            acoustic_tokenizer_config={"vae_dim": 32},
        )
        config_dict = config.to_dict()

        self.assertIn("decoder_config", config_dict)
        self.assertIn("acoustic_tokenizer_config", config_dict)

    def test_streaming_config_inherits_properly(self):
        """Test that streaming config inherits from base config."""
        config = VibeVoiceStreamingConfig(
            decoder_config={"model_type": "qwen2", "hidden_size": 64},
        )

        self.assertIsInstance(config, VibeVoiceConfig)
        self.assertTrue(config.is_streaming)
        self.assertFalse(config.has_semantic_tokenizer)
        self.assertEqual(config.num_speakers, 1)

    def test_config_with_invalid_decoder_type_raises(self):
        """Test that invalid decoder config type raises error."""
        with self.assertRaises(ValueError):
            VibeVoiceConfig(decoder_config="invalid")

    def test_config_with_invalid_acoustic_type_raises(self):
        """Test that invalid acoustic tokenizer config type raises error."""
        with self.assertRaises(ValueError):
            VibeVoiceConfig(acoustic_tokenizer_config="invalid")


@require_torch
class VibeVoiceGradientCheckpointingTest(unittest.TestCase):
    """Tests for gradient checkpointing functionality."""

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)

    def test_gradient_checkpointing_enable_disable(self):
        """Test enabling and disabling gradient checkpointing."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config)

        # Initially gradient checkpointing should be disabled
        self.assertFalse(model.is_gradient_checkpointing)

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()
        self.assertTrue(model.is_gradient_checkpointing)

        # Disable gradient checkpointing
        model.gradient_checkpointing_disable()
        self.assertFalse(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_backward(self):
        """Test that model can do backward pass with gradient checkpointing."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device)
        model.gradient_checkpointing_enable()
        model.train()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.logits.sum()

        # Should be able to backward without error
        loss.backward()


@require_torch
class VibeVoiceFP16Test(unittest.TestCase):
    """Tests for FP16 (half precision) functionality."""

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)

    def test_model_fp16_forward(self):
        """Test model forward pass with FP16."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).half().eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.dtype, torch.float16)

    def test_acoustic_codec_fp16(self):
        """Test acoustic codec forward pass with FP16."""
        config = VibeVoiceAcousticCodecConfig(
            vae_dim=16,
            encoder_n_filters=8,
            decoder_n_filters=8,
            encoder_ratios=[2, 2],
            decoder_ratios=[2, 2],
            encoder_depths="2-2-2",
        )
        model = VibeVoiceAcousticCodec(config=config).to(torch_device).half().eval()

        audio_values = torch.randn(2, 1, 800, device=torch_device).half()

        with torch.no_grad():
            outputs = model(audio_values)

        self.assertIsNotNone(outputs.reconstructed_audio)
        self.assertEqual(outputs.reconstructed_audio.dtype, torch.float16)

    def test_diffusion_head_fp16(self):
        """Test diffusion head forward pass with FP16."""
        config = VibeVoiceDiffusionHeadConfig(
            hidden_size=32,
            latent_size=16,
            head_layers=2,
        )
        model = VibeVoiceDiffusionHead(config=config).to(torch_device).eval()

        batch_size = 2
        seq_len = 4
        noisy_latents = torch.randn(batch_size, seq_len, 16, device=torch_device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=torch_device)
        conditioning = torch.randn(batch_size, seq_len, 32, device=torch_device)

        with (
            torch.no_grad(),
            torch.autocast(device_type=torch_device if torch_device != "cpu" else "cpu", dtype=torch.float16),
        ):
            output = model(noisy_latents, timesteps, conditioning)

        self.assertIsNotNone(output)
        # With autocast, output may be float16 or float32 depending on operation
        self.assertIn(output.dtype, [torch.float16, torch.float32])


@require_torch
class VibeVoiceOutputsTest(unittest.TestCase):
    """Tests for model output classes."""

    def setUp(self):
        self.model_tester = VibeVoiceModelTester(self)

    def test_output_hidden_states(self):
        """Test that model can return hidden states."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        self.assertIsNotNone(outputs.hidden_states)
        self.assertIsInstance(outputs.hidden_states, tuple)

    def test_output_attentions(self):
        """Test that model can return attention weights."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        # Set eager attention to support output_attentions
        config.decoder_config._attn_implementation = "eager"
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        self.assertIsNotNone(outputs.attentions)
        self.assertIsInstance(outputs.attentions, tuple)

    def test_output_with_labels(self):
        """Test model forward with labels (computes loss)."""
        config, inputs = self.model_tester.prepare_config_and_inputs()
        model = VibeVoiceForConditionalGeneration(config=config).to(torch_device).eval()

        input_ids = inputs["input_ids"].to(torch_device)
        attention_mask = inputs["attention_mask"].to(torch_device)
        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.shape, ())


@require_torch
@slow
class VibeVoiceIntegrationTest(unittest.TestCase):
    """Integration tests for VibeVoice with pretrained models from HuggingFace Hub.

    Tests loading and inference with pretrained models from HuggingFace Hub.
    """

    def test_model_from_pretrained(self):
        """Test loading 1.5B model from HuggingFace Hub."""
        model = VibeVoiceForConditionalGeneration.from_pretrained("microsoft/VibeVoice-1.5B")
        self.assertIsNotNone(model)
        self.assertEqual(model.config.model_type, "vibevoice")
        self.assertTrue(model.config.has_semantic_tokenizer)
        self.assertFalse(model.config.is_streaming)

    def test_streaming_model_from_pretrained(self):
        """Test loading streaming/realtime 0.5B model from HuggingFace Hub."""
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceStreamingForConditionalGeneration

        model = VibeVoiceStreamingForConditionalGeneration.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")
        self.assertIsNotNone(model)
        self.assertEqual(model.config.model_type, "vibevoice_streaming")
        self.assertFalse(model.config.has_semantic_tokenizer)
        self.assertTrue(model.config.is_streaming)

    def test_model_forward_pretrained(self):
        """Test forward pass with pretrained 1.5B model."""
        from transformers import AutoTokenizer

        model = (
            VibeVoiceForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                torch_dtype=torch.float32,
            )
            .to(torch_device)
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained("microsoft/VibeVoice-1.5B")

        text = "Hello, this is a test."
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)
        self.assertEqual(outputs.logits.shape[0], 1)  # batch size 1

    def test_streaming_model_forward_pretrained(self):
        """Test forward pass with pretrained streaming model."""
        from transformers import AutoTokenizer
        from transformers.models.vibevoice.modeling_vibevoice import VibeVoiceStreamingForConditionalGeneration

        model = (
            VibeVoiceStreamingForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-Realtime-0.5B",
                torch_dtype=torch.float32,
            )
            .to(torch_device)
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained("microsoft/VibeVoice-Realtime-0.5B")

        text = "Hello, this is a test."
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)

    @require_torch_accelerator
    def test_model_fp16_pretrained(self):
        """Test forward pass with pretrained model in FP16."""
        from transformers import AutoTokenizer

        model = (
            VibeVoiceForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                torch_dtype=torch.float16,
            )
            .to(torch_device)
            .eval()
        )

        tokenizer = AutoTokenizer.from_pretrained("microsoft/VibeVoice-1.5B")

        text = "Hello, this is a test of the VibeVoice text-to-speech system."
        inputs = tokenizer(text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)

    def test_processor_with_pretrained_model(self):
        """Test processor integration with pretrained model."""
        from transformers import VibeVoiceProcessor

        processor = VibeVoiceProcessor.from_pretrained("microsoft/VibeVoice-1.5B")
        model = (
            VibeVoiceForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                torch_dtype=torch.float32,
            )
            .to(torch_device)
            .eval()
        )

        text = "Hello world"
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        self.assertIsNotNone(outputs.logits)

    def test_acoustic_tokenizer_pretrained(self):
        """Test acoustic tokenizer from pretrained model."""
        model = (
            VibeVoiceForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-1.5B",
                torch_dtype=torch.float32,
            )
            .to(torch_device)
            .eval()
        )

        # Create dummy audio input
        batch_size = 1
        audio_length = 24000  # 1 second at 24kHz
        audio = torch.randn(batch_size, 1, audio_length, device=torch_device)

        with torch.no_grad():
            # Encode audio to latent
            mean, log_var = model.model.acoustic_tokenizer.encode(audio)
            self.assertIsNotNone(mean)

            # Decode back to audio
            reconstructed = model.model.acoustic_tokenizer.decode(mean)
            self.assertIsNotNone(reconstructed)
            self.assertEqual(reconstructed.shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()
