# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3-TTS model."""

import copy
import sys
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import cleanup, require_torch, require_torch_accelerator, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

if is_torch_available():
    import torch
    from transformers import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
        Qwen3TTSSpeakerEncoderConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTokenizerV1Config,
        Qwen3TTSTokenizerV2Config,
    )
    from transformers.models.qwen3_tts.modeling_qwen3_tts import (
        Qwen3TTSTokenizerV1Decoder,
        Qwen3TTSTokenizerV1DecoderBigVGANModel,
        Qwen3TTSTokenizerV1DecoderDiTModel,
        Qwen3TTSTokenizerV2Model,
        Qwen3TTSTokenizerV2TransformerModel,
    )


class Qwen3TTSTalkerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=16,
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_code_groups=4,
        intermediate_size=128,
        is_training=True,
        text_vocab_size=64,
        text_hidden_size=32,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_code_groups = num_code_groups
        self.intermediate_size = intermediate_size
        self.is_training = is_training
        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size

    def get_config(self):
        return Qwen3TTSTalkerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            use_sliding_window=False,
            num_code_groups=self.num_code_groups,
            text_vocab_size=self.text_vocab_size,
            text_hidden_size=self.text_hidden_size,
            code_predictor_config={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "max_position_embeddings": 2048,
                "num_code_groups": self.num_code_groups,
                "use_sliding_window": False,
                "pad_token_id": None,
            },
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Qwen3TTSTalkerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3TTSTalkerModel,) if is_torch_available() else ()
    all_generative_model_classes = ()
    pipeline_model_mapping = {}
    test_headmasking = False
    test_pruning = False
    test_resize_embeddings = False
    test_resize_embeddings_untied = False

    def setUp(self):
        self.model_tester = Qwen3TTSTalkerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3TTSTalkerConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            inputs_dict["labels"] = ids_tensor(
                [self.model_tester.batch_size, self.model_tester.seq_length],
                self.model_tester.vocab_size,
            )
        return inputs_dict

    @unittest.skip(reason="Qwen3TTSTalker codec_embedding differs from standard text embed_tokens.")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="Qwen3TTSTalker base model returns no loss.")
    def test_training(self):
        pass

    @unittest.skip(reason="Qwen3TTSTalker base model returns no loss.")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Qwen3TTSTalker base model returns no loss.")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="Qwen3TTSTalker base model returns no loss.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Qwen3TTSTalker base model returns no loss.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skipIf(sys.platform == "win32", "safetensors file locking not supported on Windows.")
    def test_save_load(self):
        super().test_save_load()

    def test_conditional_generation_forward(self):
        """Test ForConditionalGeneration prefill path with inputs_embeds."""
        config = self.model_tester.get_config()
        model = Qwen3TTSTalkerForConditionalGeneration(config).to(torch_device)
        model.eval()
        inputs_embeds = floats_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size]
        )
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds)
        self.assertEqual(
            outputs.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.vocab_size),
        )

    def test_conditional_generation_with_labels(self):
        config = self.model_tester.get_config()
        model = Qwen3TTSTalkerForConditionalGeneration(config)
        model.train()
        inputs_embeds = floats_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size]
        )
        labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.vocab_size)
        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        self.assertIsNotNone(outputs.loss)

    def test_sdpa_attention_implementation(self):
        """Talker model runs correctly under the SDPA attention backend."""
        config = self.model_tester.get_config()
        config._attn_implementation = "sdpa"
        model = Qwen3TTSTalkerModel(config).to(torch_device)
        model.eval()
        _, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        with torch.no_grad():
            outputs = model(**input_dict)
        self.assertEqual(
            outputs.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    def test_output_hidden_states(self):
        """Talker model returns all intermediate hidden states when requested."""
        config = self.model_tester.get_config()
        model = Qwen3TTSTalkerModel(config).to(torch_device)
        model.eval()
        _, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        with torch.no_grad():
            outputs = model(**input_dict, output_hidden_states=True)
        self.assertIsNotNone(outputs.hidden_states)
        # num_hidden_layers intermediate + 1 final = num_hidden_layers + 1
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        for hs in outputs.hidden_states:
            self.assertEqual(
                hs.shape,
                (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
            )

    def test_past_key_values_caching(self):
        """Prefill with use_cache=True produces past_key_values and same output shape."""
        config = self.model_tester.get_config()
        model = Qwen3TTSTalkerModel(config).to(torch_device)
        model.eval()
        _, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        with torch.no_grad():
            out_no_cache = model(**input_dict, use_cache=False)
            out_cached = model(**input_dict, use_cache=True)
        self.assertIsNotNone(out_cached.past_key_values)
        self.assertEqual(out_no_cache.last_hidden_state.shape, out_cached.last_hidden_state.shape)


@require_torch
class Qwen3TTSTokenizerModelTest(unittest.TestCase):
    """Test V1 and V2 speech tokenizer sub-models."""

    def _get_v2_config(self):
        return Qwen3TTSTokenizerV2Config(
            encoder_config={
                "audio_channels": 1,
                "chunk_in_sec": None,
                "hidden_size": 32,
                "num_filters": 8,
                "num_residual_layers": 1,
                "upsampling_ratios": [8, 4],
                "codebook_size": 64,
                "vector_quantization_hidden_dimension": 64,
                "upsample_groups": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "sliding_window": 4,
                "codebook_dim": 64,
                "use_cache": False,
            },
            decoder_config={
                "codebook_size": 64,
                "hidden_size": 64,
                "latent_dim": 64,
                "max_position_embeddings": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_quantizers": 4,
                "sliding_window": 8,
                "codebook_dim": 32,
                "decoder_dim": 64,
                "upsample_rates": (2, 2, 2, 2),
                "upsampling_ratios": (2, 2),
            },
            encoder_valid_num_quantizers=4,
            input_sample_rate=24000,
            output_sample_rate=24000,
            decode_upsample_rate=16,
            encode_downsample_rate=16,
        )

    # ── V2 tests ──

    def test_v2_decoder_transformer_forward(self):
        decoder_config = self._get_v2_config().decoder_config
        model = Qwen3TTSTokenizerV2TransformerModel(decoder_config).to(torch_device)
        model.eval()
        hidden_states = torch.randn(2, 10, decoder_config.hidden_size, device=torch_device)
        with torch.no_grad():
            output = model(inputs_embeds=hidden_states)
        self.assertEqual(output.last_hidden_state.shape, (2, 10, decoder_config.hidden_size))

    def test_v2_decode(self):
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 4, 4), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=True)
        self.assertEqual(len(output.audio_values), 1)
        self.assertEqual(output.audio_values[0].dim(), 1)

    def test_v2_save_load(self):
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 4, 4), device=torch_device)
        with torch.no_grad():
            output_before = model.decode(audio_codes, return_dict=True).audio_values[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Qwen3TTSTokenizerV2Model.from_pretrained(tmpdir).to(torch_device)
        loaded.eval()
        with torch.no_grad():
            output_after = loaded.decode(audio_codes, return_dict=True).audio_values[0]
        self.assertTrue(torch.allclose(output_before, output_after))

    # ── V1 tests ──

    def test_v1_decoder_forward(self):
        config = Qwen3TTSTokenizerV1Config(
            encoder_config={"n_mels": 64, "n_layer": 2},
            decoder_config={"dit_config": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}},
        ).decoder_config
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        codes = torch.randint(0, 512, (2, 50), device=torch_device)
        conditioning = torch.randn(2, 192, device=torch_device)
        reference_mel = torch.randn(2, 300, 80, device=torch_device)
        with torch.no_grad():
            outputs = model(codes, conditioning, reference_mel, num_steps=2)
        self.assertEqual(outputs.shape[0], 2)

    @unittest.skipIf(sys.platform == "win32", "safetensors file locking not supported on Windows.")
    def test_v1_decoder_save_load(self):
        """V1 decoder produces identical output after save/load."""
        config = Qwen3TTSTokenizerV1Config(
            encoder_config={"n_mels": 64, "n_layer": 2},
            decoder_config={"dit_config": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4}},
        ).decoder_config
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        codes = torch.randint(0, 512, (1, 50), device=torch_device)
        conditioning = torch.randn(1, 192, device=torch_device)
        reference_mel = torch.randn(1, 300, 80, device=torch_device)
        with torch.no_grad():
            output_before = model(codes, conditioning, reference_mel, num_steps=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            loaded = Qwen3TTSTokenizerV1Decoder.from_pretrained(tmpdir).to(torch_device)
        loaded.eval()
        with torch.no_grad():
            output_after = loaded(codes, conditioning, reference_mel, num_steps=2)
        self.assertTrue(torch.allclose(output_before, output_after))

    def test_v2_batch_decode(self):
        """V2 tokenizer decodes a batch of two items producing one waveform per item."""
        config = self._get_v2_config()
        num_q = config.decoder_config.num_quantizers
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        # decode expects shape (batch, seq_len, num_quantizers)
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (2, 4, num_q), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=True)
        self.assertEqual(len(output.audio_values), 2)
        for wav in output.audio_values:
            self.assertEqual(wav.dim(), 1)

    def test_v2_output_length_scales_with_codes(self):
        """V2 decoder output audio length doubles when the code sequence doubles."""
        config = self._get_v2_config()
        num_q = config.decoder_config.num_quantizers
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        # decode expects shape (batch, seq_len, num_quantizers)
        codes_4 = torch.randint(1, config.decoder_config.codebook_size, (1, 4, num_q), device=torch_device)
        codes_8 = torch.randint(1, config.decoder_config.codebook_size, (1, 8, num_q), device=torch_device)
        with torch.no_grad():
            out_4 = model.decode(codes_4, return_dict=True).audio_values[0]
            out_8 = model.decode(codes_8, return_dict=True).audio_values[0]
        self.assertEqual(len(out_8), 2 * len(out_4))


@require_torch
class Qwen3TTSTokenizerV2Test(unittest.TestCase):
    """Comprehensive tests for the V2 (12Hz) speech tokenizer."""

    def _get_v2_config(self):
        return Qwen3TTSTokenizerV2Config(
            encoder_config={
                "audio_channels": 1,
                "chunk_in_sec": None,
                "hidden_size": 32,
                "num_filters": 8,
                "num_residual_layers": 1,
                "upsampling_ratios": [8, 4],
                "codebook_size": 64,
                "vector_quantization_hidden_dimension": 64,
                "upsample_groups": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "sliding_window": 4,
                "codebook_dim": 64,
                "use_cache": False,
            },
            decoder_config={
                "codebook_size": 64,
                "hidden_size": 64,
                "latent_dim": 64,
                "max_position_embeddings": 256,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_quantizers": 4,
                "sliding_window": 8,
                "codebook_dim": 32,
                "decoder_dim": 64,
                "upsample_rates": (2, 2, 2, 2),
                "upsampling_ratios": (2, 2),
            },
            encoder_valid_num_quantizers=4,
            input_sample_rate=24000,
            output_sample_rate=24000,
            decode_upsample_rate=16,
            encode_downsample_rate=16,
        )

    def test_v2_decode_output_is_valid_audio(self):
        """V2 decode output is float, finite, and within [-1, 1]."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 6, num_q), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=True)
        wav = output.audio_values[0]
        self.assertTrue(wav.is_floating_point())
        self.assertTrue(torch.isfinite(wav).all())
        self.assertLessEqual(wav.max().item(), 1.0)
        self.assertGreaterEqual(wav.min().item(), -1.0)

    def test_v2_decode_audio_length_matches_upsample_rate(self):
        """V2 decoded audio length equals codes_length * decode_upsample_rate."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        codes_len = 5
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, codes_len, num_q), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=True)
        wav = output.audio_values[0]
        expected_length = codes_len * config.decode_upsample_rate
        self.assertEqual(len(wav), expected_length)

    def test_v2_decode_no_nans(self):
        """V2 decode does not produce NaN values."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 8, num_q), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=True)
        wav = output.audio_values[0]
        self.assertFalse(torch.isnan(wav).any())

    def test_v2_decode_deterministic(self):
        """V2 decode produces identical output for the same input."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 4, num_q), device=torch_device)
        with torch.no_grad():
            out1 = model.decode(audio_codes, return_dict=True).audio_values[0]
            out2 = model.decode(audio_codes, return_dict=True).audio_values[0]
        self.assertTrue(torch.equal(out1, out2))

    def test_v2_decode_returns_tuple_when_return_dict_false(self):
        """V2 decode returns a tuple when return_dict=False."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        audio_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 4, num_q), device=torch_device)
        with torch.no_grad():
            output = model.decode(audio_codes, return_dict=False)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output[0]), 1)

    def test_v2_config_roundtrip(self):
        """V2 config saves and loads with identical values."""
        config = self._get_v2_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = Qwen3TTSTokenizerV2Config.from_pretrained(tmpdir)
        self.assertEqual(config.decoder_config.num_quantizers, loaded.decoder_config.num_quantizers)
        self.assertEqual(config.decode_upsample_rate, loaded.decode_upsample_rate)
        self.assertEqual(config.encoder_valid_num_quantizers, loaded.encoder_valid_num_quantizers)

    def test_v2_decode_wrong_num_quantizers_raises(self):
        """V2 decode raises when codes have the wrong number of quantizers."""
        config = self._get_v2_config()
        model = Qwen3TTSTokenizerV2Model(config).to(torch_device)
        model.eval()
        num_q = config.decoder_config.num_quantizers
        wrong_q = num_q + 2
        bad_codes = torch.randint(1, config.decoder_config.codebook_size, (1, 4, wrong_q), device=torch_device)
        with self.assertRaises((ValueError, RuntimeError)):
            with torch.no_grad():
                model.decode(bad_codes, return_dict=True)


@require_torch
class Qwen3TTSTokenizerV1Test(unittest.TestCase):
    """Comprehensive tests for the V1 (25Hz) speech tokenizer."""

    def _get_v1_decoder_config(self):
        return Qwen3TTSTokenizerV1Config(
            encoder_config={"n_mels": 64, "n_layer": 2},
            decoder_config={
                "dit_config": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4},
            },
        ).decoder_config

    def _get_v1_config(self):
        return Qwen3TTSTokenizerV1Config(
            encoder_config={
                "n_mels": 64,
                "n_layer": 2,
                "n_state": 64,
                "n_head": 4,
                "output_dim": 64,
                "audio_vq_type": "GRVQ",
                "audio_vq_codebook_size": 64,
                "audio_vq_codebook_dim": 64,
            },
            decoder_config={
                "dit_config": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4},
            },
            decode_upsample_rate=200,
            encode_downsample_rate=200,
        )

    def test_v1_decoder_output_shape(self):
        """V1 decoder produces waveform with batch dimension preserved."""
        config = self._get_v1_decoder_config()
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        batch_size = 3
        codes = torch.randint(0, 512, (batch_size, 50), device=torch_device)
        conditioning = torch.randn(batch_size, 192, device=torch_device)
        reference_mel = torch.randn(batch_size, 300, 80, device=torch_device)
        with torch.no_grad():
            outputs = model(codes, conditioning, reference_mel, num_steps=2)
        self.assertEqual(outputs.shape[0], batch_size)

    def test_v1_decoder_output_is_valid_waveform(self):
        """V1 decoder output is a finite, non-empty waveform clamped to [-1, 1]."""
        config = self._get_v1_decoder_config()
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        codes = torch.randint(0, 512, (1, 50), device=torch_device)
        conditioning = torch.randn(1, 192, device=torch_device)
        reference_mel = torch.randn(1, 300, 80, device=torch_device)
        with torch.no_grad():
            waveform = model(codes, conditioning, reference_mel, num_steps=2)
        self.assertGreater(waveform.numel(), 0)
        self.assertTrue(torch.isfinite(waveform).all())
        self.assertLessEqual(waveform.max().item(), 1.0)
        self.assertGreaterEqual(waveform.min().item(), -1.0)

    def test_v1_decoder_output_no_nans(self):
        """V1 decoder output contains no NaN values."""
        config = self._get_v1_decoder_config()
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        codes = torch.randint(0, 512, (1, 50), device=torch_device)
        conditioning = torch.randn(1, 192, device=torch_device)
        reference_mel = torch.randn(1, 300, 80, device=torch_device)
        with torch.no_grad():
            outputs = model(codes, conditioning, reference_mel, num_steps=2)
        self.assertFalse(torch.isnan(outputs).any())
        self.assertTrue(torch.isfinite(outputs).all())

    def test_v1_dit_forward(self):
        """V1 DiT model forward pass produces output of the correct shape."""
        config = self._get_v1_decoder_config()
        dit = Qwen3TTSTokenizerV1DecoderDiTModel(config.dit_config).to(torch_device)
        dit.eval()
        batch_size = 2
        mel_dim = config.dit_config.mel_dim
        repeats = config.dit_config.repeats
        code_len = 20
        seq_len = code_len * repeats
        hidden_states = torch.randn(batch_size, seq_len, mel_dim, device=torch_device)
        speaker_embedding = torch.randn(batch_size, seq_len, 192, device=torch_device)
        condition_vector = torch.randn(batch_size, 300, mel_dim, device=torch_device)
        quantized_code = torch.randint(0, 512, (batch_size, code_len), device=torch_device)
        time_step = torch.tensor(0.5, device=torch_device)
        with torch.no_grad():
            output = dit(
                hidden_states=hidden_states,
                speaker_embedding=speaker_embedding,
                condition_vector=condition_vector,
                quantized_code=quantized_code,
                time_step=time_step,
            )
        # DiT doubles the batch via classifier-free guidance (apply_cfg=True by default)
        self.assertEqual(output.shape, (batch_size * 2, seq_len, mel_dim))

    def test_v1_bigvgan_forward(self):
        """V1 BigVGAN vocoder converts mel spectrogram to waveform."""
        bigvgan_config = Qwen3TTSTokenizerV1Config(
            decoder_config={
                "dit_config": {"hidden_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4},
            },
        ).decoder_config.bigvgan_config
        model = Qwen3TTSTokenizerV1DecoderBigVGANModel(bigvgan_config).to(torch_device)
        model.eval()
        batch_size = 2
        mel_len = 50
        mel_dim = bigvgan_config.mel_dim
        # BigVGAN conv_pre expects channels-first: (batch, mel_dim, mel_len)
        mel_input = torch.randn(batch_size, mel_dim, mel_len, device=torch_device)
        with torch.no_grad():
            waveform = model(mel_input)
        # BigVGAN squeezes and moves to CPU in its forward
        self.assertGreater(waveform.numel(), 0)
        # Waveform length is mel_len * product(upsample_rates)
        expected_upsample = 1
        for rate in bigvgan_config.upsample_rates:
            expected_upsample *= rate
        expected_samples = mel_len * expected_upsample
        # Output is squeezed, so shape depends on batch_size
        self.assertEqual(waveform.shape, (batch_size, expected_samples))

    def test_v1_config_roundtrip(self):
        """V1 config saves and loads with identical values."""
        config = self._get_v1_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = Qwen3TTSTokenizerV1Config.from_pretrained(tmpdir)
        self.assertEqual(config.decoder_config.dit_config.hidden_size, loaded.decoder_config.dit_config.hidden_size)
        self.assertEqual(config.decode_upsample_rate, loaded.decode_upsample_rate)
        self.assertEqual(config.encoder_config.n_mels, loaded.encoder_config.n_mels)

    def test_v1_decoder_longer_codes_produce_longer_audio(self):
        """V1 decoder output audio length grows proportionally with code sequence length."""
        config = self._get_v1_decoder_config()
        model = Qwen3TTSTokenizerV1Decoder(config).to(torch_device)
        model.eval()
        conditioning = torch.randn(1, 192, device=torch_device)
        reference_mel = torch.randn(1, 300, 80, device=torch_device)
        codes_short = torch.randint(0, 512, (1, 25), device=torch_device)
        codes_long = torch.randint(0, 512, (1, 50), device=torch_device)
        with torch.no_grad():
            out_short = model(codes_short, conditioning, reference_mel, num_steps=2)
            out_long = model(codes_long, conditioning, reference_mel, num_steps=2)
        self.assertGreater(out_long.numel(), out_short.numel())


@require_torch
class Qwen3TTSTopLevelConfigTest(unittest.TestCase):
    """Tests for Qwen3TTSConfig and its sub-configs."""

    def test_default_sub_configs_are_created(self):
        """Qwen3TTSConfig always instantiates its sub-configs even when not provided."""
        config = Qwen3TTSConfig()
        self.assertIsInstance(config.talker_config, Qwen3TTSTalkerConfig)
        self.assertIsInstance(config.speaker_encoder_config, Qwen3TTSSpeakerEncoderConfig)
        self.assertIsInstance(config.talker_config.code_predictor_config, Qwen3TTSTalkerCodePredictorConfig)

    def test_config_model_type(self):
        """model_type strings are set correctly at every level."""
        config = Qwen3TTSConfig()
        self.assertEqual(config.model_type, "qwen3_tts")
        self.assertEqual(config.talker_config.model_type, "qwen3_tts_talker")
        self.assertEqual(config.speaker_encoder_config.model_type, "qwen3_tts_speaker_encoder")

    def test_config_roundtrip(self):
        """Qwen3TTSConfig serialises and deserialises all sub-configs correctly."""
        config = Qwen3TTSConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = Qwen3TTSConfig.from_pretrained(tmpdir)
        self.assertEqual(config.talker_config.hidden_size, loaded.talker_config.hidden_size)
        self.assertEqual(config.talker_config.num_code_groups, loaded.talker_config.num_code_groups)
        self.assertEqual(config.speaker_encoder_config.enc_dim, loaded.speaker_encoder_config.enc_dim)
        self.assertEqual(config.tts_pad_token_id, loaded.tts_pad_token_id)
        self.assertEqual(config.tts_bos_token_id, loaded.tts_bos_token_id)
        self.assertEqual(config.tts_eos_token_id, loaded.tts_eos_token_id)

    def test_config_custom_talker_params(self):
        """Custom talker parameters are stored and accessible on Qwen3TTSConfig."""
        config = Qwen3TTSConfig(
            talker_config={"hidden_size": 512, "num_code_groups": 16},
            tts_pad_token_id=100,
        )
        self.assertEqual(config.talker_config.hidden_size, 512)
        self.assertEqual(config.talker_config.num_code_groups, 16)
        self.assertEqual(config.tts_pad_token_id, 100)

    def test_talker_code_predictor_config_defaults(self):
        """Qwen3TTSTalkerCodePredictorConfig exposes expected default values."""
        cfg = Qwen3TTSTalkerCodePredictorConfig()
        self.assertEqual(cfg.vocab_size, 2048)
        self.assertEqual(cfg.num_code_groups, 32)
        self.assertEqual(cfg.model_type, "qwen3_tts_talker_code_predictor")

    def test_speaker_encoder_config_defaults(self):
        """Qwen3TTSSpeakerEncoderConfig exposes expected default values."""
        cfg = Qwen3TTSSpeakerEncoderConfig()
        self.assertEqual(cfg.enc_dim, 1024)
        self.assertEqual(cfg.sample_rate, 24000)
        self.assertEqual(cfg.model_type, "qwen3_tts_speaker_encoder")


@require_torch
class Qwen3TTSIntegrationTest(unittest.TestCase):
    """Integration tests for Qwen3-TTS (require real weights, run with --slow)."""

    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_model_and_processor(self):
        processor = Qwen3TTSProcessor.from_pretrained(self.model_id)
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.model_id, device_map=torch_device, torch_dtype=torch.bfloat16
        )
        model.eval()
        return model, processor

    @slow
    @require_torch_accelerator
    def test_small_model_integration_text_to_codes(self):
        """Text to acoustic codes; verifies exact deterministic token generation."""
        model, processor = self._load_model_and_processor()

        text = "Hello, how are you doing today?"
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, _ = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=100,
        )

        self.assertEqual(len(codes_list), 1)
        codes = codes_list[0]

        # Verify full output matches original model implementation
        # fmt: off
        EXPECTED_CODES = torch.tensor(
            [
        [
            391,
            597,
            132,
            143,
            783,
            1051,
            1463,
            609,
            73,
            661,
            357,
            574,
            1234,
            364,
            1098,
            57
        ],
        [
            1517,
            1334,
            610,
            1746,
            487,
            1387,
            1542,
            584,
            1420,
            1816,
            906,
            95,
            1389,
            1160,
            236,
            247
        ],
        [
            1032,
            1412,
            130,
            793,
            289,
            233,
            1772,
            1555,
            130,
            281,
            1991,
            925,
            696,
            133,
            77,
            47
        ],
        [
            1032,
            667,
            848,
            859,
            1981,
            2000,
            50,
            767,
            567,
            1816,
            297,
            1354,
            277,
            351,
            681,
            279
        ],
        [
            1032,
            1209,
            1967,
            1975,
            440,
            1826,
            1542,
            512,
            231,
            1031,
            445,
            1507,
            1263,
            402,
            240,
            137
        ],
        [
            2030,
            1361,
            396,
            828,
            440,
            813,
            1008,
            512,
            1370,
            1183,
            1164,
            1267,
            886,
            1174,
            51,
            799
        ],
        [
            2030,
            1209,
            563,
            639,
            223,
            1210,
            731,
            512,
            1236,
            1253,
            1226,
            706,
            1292,
            1539,
            1681,
            270
        ],
        [
            2030,
            1361,
            1208,
            1759,
            923,
            1520,
            387,
            286,
            348,
            500,
            656,
            354,
            673,
            90,
            1398,
            1204
        ],
        [
            51,
            605,
            787,
            761,
            110,
            1343,
            778,
            1085,
            12,
            225,
            341,
            1104,
            312,
            477,
            623,
            1016
        ],
        [
            327,
            1209,
            787,
            1907,
            485,
            198,
            1706,
            544,
            74,
            533,
            1993,
            1124,
            1159,
            1195,
            466,
            969
        ],
        [
            327,
            1209,
            787,
            1514,
            1424,
            1210,
            879,
            1375,
            174,
            1116,
            615,
            63,
            42,
            1003,
            537,
            1750
        ],
        [
            85,
            1209,
            787,
            1095,
            939,
            909,
            676,
            424,
            301,
            1473,
            255,
            1532,
            1769,
            477,
            827,
            536
        ],
        [
            781,
            1038,
            251,
            1731,
            1861,
            670,
            1060,
            1457,
            303,
            103,
            273,
            582,
            266,
            2019,
            74,
            524
        ],
        [
            882,
            1039,
            1928,
            598,
            176,
            495,
            2038,
            706,
            1881,
            1294,
            943,
            1755,
            1702,
            792,
            148,
            925
        ],
        [
            882,
            1481,
            1730,
            1566,
            1637,
            937,
            310,
            424,
            1854,
            266,
            1126,
            1466,
            150,
            2040,
            20,
            363
        ],
        [
            882,
            706,
            1063,
            250,
            1172,
            1002,
            1975,
            965,
            174,
            1794,
            301,
            1976,
            717,
            344,
            1594,
            116
        ],
        [
            882,
            910,
            1955,
            1954,
            1276,
            1587,
            804,
            1182,
            1311,
            1042,
            1064,
            178,
            1555,
            1422,
            1459,
            1817
        ],
        [
            1128,
            919,
            928,
            178,
            1658,
            1425,
            1292,
            1148,
            859,
            438,
            1042,
            1047,
            532,
            926,
            1412,
            127
        ],
        [
            1501,
            919,
            2025,
            222,
            1501,
            1002,
            259,
            61,
            405,
            1314,
            1466,
            796,
            206,
            438,
            1193,
            447
        ],
        [
            1501,
            919,
            2025,
            1327,
            1047,
            330,
            1759,
            1988,
            1904,
            956,
            743,
            1382,
            527,
            1032,
            85,
            436
        ],
        [
            366,
            910,
            1972,
            890,
            385,
            1221,
            1062,
            486,
            794,
            104,
            288,
            1377,
            996,
            993,
            73,
            385
        ],
        [
            366,
            910,
            0,
            890,
            625,
            258,
            350,
            824,
            644,
            1221,
            288,
            1331,
            527,
            661,
            1431,
            1978
        ],
        [
            1318,
            160,
            1319,
            1278,
            281,
            1199,
            485,
            1344,
            299,
            379,
            1951,
            1166,
            1124,
            1273,
            255,
            733
        ],
        [
            1894,
            1838,
            1861,
            1721,
            1088,
            351,
            1494,
            687,
            303,
            631,
            1839,
            1474,
            526,
            1355,
            289,
            622
        ],
        [
            1894,
            844,
            118,
            1027,
            1086,
            150,
            109,
            1288,
            935,
            249,
            1853,
            1850,
            705,
            1112,
            272,
            1759
        ],
        [
            1894,
            614,
            1666,
            69,
            54,
            151,
            1528,
            1853,
            964,
            1758,
            120,
            46,
            156,
            1762,
            1737,
            1568
        ],
        [
            1894,
            614,
            1666,
            420,
            197,
            1203,
            479,
            1275,
            1167,
            1440,
            447,
            682,
            990,
            1414,
            792,
            409
        ],
        [
            302,
            614,
            1666,
            111,
            197,
            587,
            100,
            179,
            1955,
            1555,
            879,
            258,
            22,
            320,
            98,
            1254
        ],
        [
            302,
            614,
            550,
            111,
            197,
            1189,
            1815,
            1275,
            1167,
            1176,
            310,
            524,
            899,
            153,
            20,
            1429
        ],
        [
            302,
            614,
            1666,
            111,
            197,
            208,
            662,
            870,
            935,
            249,
            271,
            154,
            298,
            2006,
            1397,
            113
        ],
        [
            302,
            1557,
            118,
            610,
            197,
            126,
            1479,
            1688,
            1681,
            1433,
            1171,
            121,
            289,
            4,
            15,
            1119
        ],
        [
            302,
            1557,
            1666,
            1219,
            1617,
            126,
            1479,
            62,
            337,
            653,
            320,
            446,
            289,
            1771,
            62,
            518
        ],
        [
            302,
            1314,
            1231,
            1089,
            824,
            86,
            663,
            323,
            32,
            636,
            872,
            490,
            1808,
            446,
            1166,
            6
        ],
        [
            1570,
            1134,
            1666,
            2018,
            227,
            86,
            398,
            711,
            1156,
            41,
            1311,
            613,
            2030,
            298,
            259,
            1796
        ],
        [
            1570,
            614,
            1666,
            679,
            211,
            176,
            63,
            1056,
            1053,
            512,
            271,
            1563,
            374,
            685,
            372,
            1552
        ],
        [
            1570,
            614,
            1666,
            420,
            211,
            721,
            1363,
            1275,
            1353,
            1440,
            37,
            1449,
            37,
            564,
            827,
            196
        ],
        [
            1570,
            614,
            1666,
            420,
            1994,
            721,
            1363,
            1275,
            339,
            1440,
            262,
            1449,
            333,
            755,
            573,
            600
        ],
        [
            1570,
            1132,
            482,
            1367,
            681,
            86,
            862,
            323,
            862,
            1421,
            133,
            668,
            1103,
            1615,
            1733,
            284
        ],
        [
            1570,
            1149,
            482,
            1219,
            1118,
            1609,
            63,
            1009,
            1372,
            1768,
            1311,
            668,
            1808,
            4,
            192,
            121
        ],
        [
            1570,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            1421,
            176,
            668,
            1103,
            752,
            1600,
            1072
        ],
        [
            1570,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            895,
            1600,
            237
        ],
        [
            1570,
            1149,
            482,
            610,
            1118,
            1609,
            63,
            1009,
            208,
            1768,
            1311,
            420,
            1567,
            4,
            117,
            695
        ],
        [
            1570,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1217,
            284
        ],
        [
            1570,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1217,
            2016
        ],
        [
            1570,
            1149,
            482,
            610,
            1118,
            146,
            1091,
            1275,
            208,
            899,
            150,
            2033,
            263,
            913,
            550,
            412
        ],
        [
            1570,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1217,
            1555
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1493,
            2016
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1493,
            2016
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1493,
            2016
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            862,
            323,
            373,
            227,
            176,
            668,
            1103,
            752,
            1217,
            2016
        ],
        [
            433,
            1338,
            2007,
            1791,
            86,
            1781,
            242,
            177,
            1580,
            2019,
            745,
            340,
            948,
            231,
            392,
            100
        ],
        [
            302,
            1132,
            1231,
            822,
            441,
            1657,
            117,
            471,
            777,
            231,
            966,
            1568,
            526,
            1021,
            199,
            1555
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            30,
            1512,
            395,
            227,
            176,
            668,
            1103,
            752,
            492,
            1100
        ],
        [
            302,
            1132,
            1231,
            1595,
            681,
            1657,
            30,
            1512,
            395,
            227,
            176,
            1875,
            1103,
            752,
            1493,
            1100
        ],
        [
            302,
            1132,
            1231,
            1595,
            917,
            86,
            862,
            323,
            1495,
            227,
            859,
            668,
            1441,
            752,
            639,
            284
        ],
        [
            302,
            1132,
            1231,
            1595,
            917,
            86,
            862,
            323,
            1495,
            227,
            859,
            668,
            598,
            752,
            639,
            284
        ],
        [
            433,
            1132,
            1231,
            822,
            441,
            1657,
            117,
            471,
            777,
            242,
            983,
            8,
            12,
            1021,
            199,
            1816
        ],
        [
            433,
            1132,
            1231,
            1595,
            1297,
            86,
            190,
            1349,
            208,
            1209,
            1447,
            668,
            1891,
            188,
            281,
            1058
        ],
        [
            433,
            1132,
            1231,
            822,
            629,
            1632,
            1445,
            1941,
            1425,
            416,
            572,
            747,
            1255,
            123,
            1865,
            221
        ],
        [
            433,
            1132,
            1231,
            822,
            629,
            1632,
            1445,
            1941,
            1425,
            416,
            1969,
            417,
            154,
            1305,
            184,
            180
        ],
        [
            433,
            1132,
            1231,
            230,
            601,
            86,
            862,
            1007,
            473,
            78,
            389,
            738,
            1446,
            1305,
            605,
            144
        ],
        [
            433,
            1132,
            1231,
            230,
            601,
            86,
            862,
            1007,
            473,
            78,
            389,
            747,
            805,
            52,
            405,
            1966
        ],
        [
            1451,
            1149,
            1244,
            1112,
            629,
            285,
            95,
            471,
            208,
            180,
            565,
            1736,
            552,
            37,
            15,
            1783
        ],
        [
            433,
            1132,
            1231,
            230,
            601,
            86,
            862,
            1007,
            473,
            78,
            859,
            738,
            526,
            1305,
            893,
            284
        ],
        [
            433,
            1132,
            366,
            1134,
            601,
            86,
            862,
            198,
            1436,
            1907,
            1247,
            1326,
            1446,
            616,
            298,
            470
        ],
        [
            433,
            1112,
            1231,
            532,
            1107,
            1222,
            1347,
            845,
            1014,
            1062,
            524,
            1943,
            282,
            156,
            343,
            571
        ],
        [
            433,
            756,
            1231,
            938,
            693,
            1366,
            514,
            845,
            247,
            965,
            382,
            14,
            698,
            389,
            192,
            1007
        ],
        [
            433,
            1260,
            366,
            457,
            1107,
            468,
            1042,
            527,
            538,
            1860,
            204,
            618,
            1722,
            1899,
            92,
            1648
        ],
        [
            2005,
            1491,
            1468,
            109,
            63,
            1765,
            705,
            122,
            1569,
            954,
            133,
            278,
            928,
            939,
            963,
            1029
        ],
        [
            1393,
            1491,
            158,
            632,
            958,
            1607,
            528,
            122,
            822,
            276,
            575,
            939,
            779,
            403,
            578,
            320
        ],
        [
            863,
            974,
            2019,
            696,
            149,
            58,
            70,
            1915,
            38,
            751,
            2009,
            707,
            973,
            21,
            1239,
            1648
        ],
        [
            1224,
            974,
            2019,
            631,
            1641,
            1181,
            384,
            1468,
            1846,
            1175,
            1177,
            455,
            628,
            1625,
            196,
            873
        ],
        [
            1224,
            1927,
            2019,
            966,
            1625,
            372,
            1130,
            787,
            1562,
            370,
            1324,
            1653,
            975,
            203,
            1064,
            137
        ],
        [
            553,
            974,
            524,
            1041,
            1078,
            344,
            84,
            710,
            338,
            760,
            715,
            1036,
            969,
            137,
            617,
            370
        ],
        [
            553,
            974,
            524,
            1218,
            55,
            1228,
            849,
            122,
            1946,
            1739,
            411,
            1229,
            853,
            264,
            1678,
            505
        ],
        [
            553,
            974,
            841,
            1105,
            1147,
            1973,
            331,
            347,
            1562,
            416,
            592,
            205,
            975,
            76,
            639,
            693
        ],
        [
            1984,
            646,
            838,
            204,
            149,
            1583,
            189,
            316,
            171,
            66,
            2009,
            773,
            806,
            510,
            1805,
            1136
        ],
        [
            1984,
            646,
            368,
            1760,
            149,
            1755,
            1794,
            1487,
            1440,
            1649,
            743,
            27,
            688,
            1866,
            341,
            1874
        ],
        [
            1984,
            646,
            841,
            168,
            1125,
            438,
            399,
            108,
            249,
            129,
            288,
            891,
            1713,
            724,
            259,
            76
        ],
        [
            1263,
            646,
            841,
            168,
            1125,
            438,
            399,
            108,
            249,
            662,
            634,
            146,
            204,
            1384,
            103,
            1894
        ],
        [
            1263,
            646,
            841,
            168,
            1854,
            674,
            970,
            1059,
            632,
            129,
            324,
            362,
            1677,
            1740,
            1008,
            1928
        ],
        [
            1281,
            646,
            841,
            168,
            695,
            2044,
            767,
            370,
            1440,
            129,
            737,
            138,
            1688,
            863,
            1473,
            233
        ],
        [
            1281,
            646,
            841,
            168,
            735,
            406,
            55,
            405,
            174,
            129,
            737,
            437,
            1243,
            513,
            1421,
            1435
        ],
        [
            1281,
            646,
            841,
            168,
            1972,
            407,
            312,
            405,
            174,
            129,
            737,
            437,
            783,
            1484,
            1229,
            108
        ],
        [
            1281,
            646,
            841,
            1702,
            1442,
            521,
            663,
            405,
            564,
            129,
            743,
            1689,
            1954,
            158,
            766,
            281
        ],
        [
            1281,
            646,
            841,
            1702,
            1442,
            521,
            663,
            405,
            288,
            254,
            645,
            1060,
            914,
            563,
            1698,
            253
        ],
        [
            1281,
            646,
            841,
            1702,
            1442,
            521,
            663,
            405,
            288,
            254,
            233,
            156,
            347,
            69,
            1032,
            615
        ],
        [
            1281,
            646,
            841,
            168,
            1972,
            33,
            2025,
            1777,
            174,
            936,
            1381,
            1780,
            330,
            333,
            596,
            9
        ],
        [
            1281,
            646,
            841,
            168,
            1972,
            209,
            34,
            96,
            214,
            836,
            623,
            1570,
            996,
            1531,
            1528,
            488
        ],
        [
            1281,
            646,
            971,
            2032,
            363,
            915,
            506,
            329,
            174,
            444,
            288,
            903,
            996,
            69,
            903,
            777
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            90,
            1794,
            2029,
            1440,
            1870,
            557,
            437,
            996,
            346,
            497,
            170
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            90,
            1794,
            2029,
            1440,
            1870,
            557,
            437,
            996,
            691,
            641,
            556
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            90,
            1794,
            2029,
            1440,
            1870,
            557,
            437,
            1713,
            1519,
            967,
            149
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            90,
            1794,
            2029,
            1440,
            1870,
            557,
            437,
            996,
            1519,
            687,
            675
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            90,
            1794,
            2029,
            1440,
            1870,
            288,
            437,
            783,
            1519,
            2039,
            185
        ],
        [
            1281,
            131,
            217,
            22,
            909,
            1543,
            849,
            739,
            632,
            1001,
            18,
            146,
            336,
            84,
            484,
            1699
        ],
        [
            1281,
            131,
            1106,
            1665,
            442,
            1228,
            849,
            1777,
            564,
            173,
            743,
            248,
            914,
            1853,
            300,
            1065
        ],
        [
            1281,
            131,
            1106,
            1665,
            442,
            1228,
            849,
            1777,
            1242,
            173,
            398,
            460,
            914,
            781,
            31,
            972
        ],
        [
            1281,
            131,
            1106,
            1665,
            550,
            1228,
            849,
            1075,
            249,
            1870,
            288,
            437,
            1713,
            1484,
            1249,
            412
        ]
    ]
        )
        # fmt: on
        torch.testing.assert_close(codes.cpu(), EXPECTED_CODES)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_batch(self):
        """Batch text to codes; verifies independent generation for multiple inputs."""
        model, processor = self._load_model_and_processor()

        texts = [
            "The weather is nice today.",
            "I enjoy listening to music.",
        ]
        inputs_0 = processor(text=texts[0], return_tensors="pt").to(torch_device)
        inputs_1 = processor(text=texts[1], return_tensors="pt").to(torch_device)

        codes_list, _ = model.generate(
            input_ids=[inputs_0["input_ids"], inputs_1["input_ids"]],
            languages=["auto", "auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=100,
        )

        self.assertEqual(len(codes_list), 2)

        # fmt: off
        EXPECTED_CODES_0 = torch.tensor(
            [
        [
            259,
            1052,
            1643,
            1357,
            346,
            747,
            559,
            1091,
            541,
            122,
            373,
            1415,
            343,
            418,
            395,
            759
        ],
        [
            415,
            574,
            651,
            1091,
            1833,
            237,
            1244,
            424,
            1828,
            811,
            624,
            612,
            1348,
            480,
            1013,
            726
        ],
        [
            1029,
            462,
            1081,
            672,
            740,
            712,
            1244,
            537,
            202,
            422,
            1851,
            1118,
            534,
            125,
            1487,
            417
        ],
        [
            663,
            1875,
            479,
            1931,
            297,
            79,
            573,
            270,
            8,
            1166,
            425,
            1127,
            528,
            1166,
            1581,
            548
        ],
        [
            695,
            1658,
            1666,
            328,
            1206,
            1214,
            948,
            270,
            8,
            1281,
            499,
            410,
            1715,
            999,
            1710,
            648
        ],
        [
            695,
            1658,
            550,
            1543,
            289,
            597,
            1038,
            91,
            1959,
            1166,
            425,
            294,
            186,
            11,
            1901,
            961
        ],
        [
            687,
            1658,
            550,
            528,
            289,
            1072,
            1614,
            91,
            1316,
            1166,
            425,
            318,
            1641,
            1564,
            861,
            570
        ],
        [
            1894,
            1658,
            550,
            527,
            1086,
            1072,
            831,
            91,
            1316,
            1166,
            341,
            232,
            1641,
            138,
            449,
            1211
        ],
        [
            1318,
            1227,
            118,
            1733,
            576,
            1072,
            990,
            1816,
            33,
            1166,
            770,
            250,
            1660,
            727,
            739,
            953
        ],
        [
            1318,
            1227,
            118,
            1733,
            576,
            1072,
            990,
            1816,
            33,
            1166,
            447,
            232,
            200,
            1131,
            739,
            124
        ],
        [
            1318,
            1227,
            118,
            1733,
            576,
            1072,
            990,
            1816,
            33,
            435,
            447,
            258,
            200,
            273,
            739,
            375
        ],
        [
            1318,
            1227,
            118,
            986,
            1086,
            1072,
            290,
            1816,
            1212,
            1748,
            447,
            232,
            462,
            369,
            114,
            449
        ],
        [
            1318,
            1227,
            1066,
            897,
            576,
            1461,
            708,
            1703,
            1910,
            107,
            701,
            854,
            154,
            921,
            10,
            1913
        ],
        [
            1318,
            1227,
            1066,
            897,
            576,
            1461,
            1078,
            1630,
            1070,
            484,
            88,
            854,
            343,
            1896,
            155,
            209
        ],
        [
            1318,
            1227,
            729,
            331,
            1362,
            293,
            1343,
            1091,
            10,
            313,
            800,
            1803,
            1099,
            1773,
            465,
            1327
        ],
        [
            948,
            807,
            367,
            897,
            411,
            95,
            277,
            1091,
            369,
            538,
            922,
            71,
            31,
            248,
            396,
            188
        ],
        [
            948,
            807,
            367,
            897,
            411,
            567,
            592,
            465,
            792,
            67,
            1493,
            1089,
            1238,
            519,
            225,
            61
        ],
        [
            605,
            807,
            1266,
            897,
            1486,
            104,
            1112,
            1168,
            727,
            1963,
            1882,
            765,
            1817,
            20,
            298,
            688
        ],
        [
            780,
            807,
            367,
            897,
            274,
            335,
            874,
            538,
            339,
            159,
            383,
            129,
            1540,
            1450,
            335,
            1362
        ],
        [
            882,
            807,
            367,
            897,
            274,
            335,
            1463,
            816,
            792,
            1320,
            517,
            1124,
            589,
            675,
            225,
            533
        ],
        [
            882,
            807,
            367,
            897,
            274,
            335,
            929,
            1379,
            151,
            15,
            2,
            325,
            677,
            70,
            49,
            1145
        ],
        [
            882,
            807,
            367,
            897,
            274,
            335,
            1463,
            1437,
            106,
            1079,
            383,
            956,
            428,
            224,
            760,
            490
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            1142,
            365,
            1173,
            935,
            688,
            1756,
            590,
            530,
            295
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            1537,
            1401,
            423,
            935,
            629,
            425,
            921,
            10,
            81
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            1796,
            159,
            1010,
            1751,
            794,
            548,
            461,
            1736
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            1066,
            415,
            1591,
            96
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            441,
            219,
            433,
            1220
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            441,
            415,
            433,
            1691
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            1066,
            415,
            190,
            81
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            195,
            984,
            190,
            1820
        ],
        [
            882,
            807,
            367,
            897,
            1141,
            335,
            1352,
            771,
            13,
            285,
            935,
            569,
            1066,
            415,
            190,
            81
        ],
        [
            125,
            807,
            1918,
            714,
            1634,
            234,
            1773,
            538,
            1804,
            484,
            369,
            1743,
            625,
            519,
            1164,
            42
        ],
        [
            125,
            1023,
            201,
            1640,
            801,
            356,
            1139,
            771,
            828,
            762,
            1699,
            273,
            1808,
            1007,
            867,
            278
        ],
        [
            125,
            1023,
            367,
            897,
            321,
            335,
            2030,
            424,
            20,
            1217,
            1742,
            131,
            1170,
            1579,
            1664,
            424
        ],
        [
            234,
            1023,
            367,
            897,
            1109,
            223,
            1352,
            1537,
            203,
            705,
            672,
            931,
            154,
            1699,
            31,
            1581
        ],
        [
            234,
            1023,
            367,
            897,
            16,
            635,
            63,
            1045,
            1309,
            1766,
            1429,
            865,
            654,
            1043,
            1110,
            450
        ],
        [
            234,
            1023,
            594,
            897,
            1876,
            1531,
            63,
            417,
            1644,
            484,
            1429,
            1232,
            97,
            800,
            208,
            1820
        ],
        [
            617,
            1023,
            594,
            897,
            1876,
            1650,
            63,
            1045,
            77,
            129,
            266,
            612,
            1031,
            415,
            1255,
            65
        ],
        [
            617,
            1023,
            594,
            897,
            1876,
            1961,
            831,
            253,
            101,
            129,
            1828,
            1313,
            129,
            1124,
            950,
            1538
        ],
        [
            617,
            1023,
            594,
            897,
            1876,
            1961,
            831,
            253,
            101,
            484,
            157,
            1313,
            524,
            992,
            595,
            1296
        ],
        [
            617,
            1023,
            594,
            897,
            1474,
            1961,
            1025,
            266,
            387,
            129,
            1828,
            1637,
            1611,
            1467,
            239,
            50
        ],
        [
            617,
            1023,
            594,
            897,
            266,
            1961,
            1120,
            462,
            387,
            409,
            3,
            1637,
            55,
            407,
            1382,
            601
        ],
        [
            617,
            1023,
            594,
            897,
            266,
            1961,
            1120,
            462,
            387,
            1967,
            176,
            1637,
            55,
            407,
            1347,
            1650
        ],
        [
            617,
            1023,
            594,
            897,
            744,
            1961,
            682,
            609,
            179,
            409,
            1884,
            1313,
            930,
            88,
            595,
            1258
        ],
        [
            234,
            1023,
            594,
            897,
            266,
            1961,
            1455,
            609,
            672,
            811,
            423,
            692,
            1094,
            407,
            205,
            981
        ],
        [
            234,
            1023,
            594,
            897,
            1865,
            1961,
            682,
            2042,
            711,
            409,
            1063,
            1089,
            827,
            1777,
            10,
            1595
        ],
        [
            234,
            1023,
            594,
            897,
            1865,
            1961,
            1514,
            609,
            1421,
            409,
            1577,
            1462,
            524,
            1779,
            241,
            862
        ],
        [
            617,
            1023,
            594,
            897,
            1865,
            1961,
            1514,
            609,
            355,
            1055,
            870,
            1313,
            725,
            321,
            357,
            129
        ],
        [
            617,
            1023,
            594,
            897,
            1865,
            1961,
            1161,
            609,
            566,
            409,
            870,
            1089,
            1863,
            992,
            1623,
            1355
        ],
        [
            617,
            1023,
            594,
            897,
            1865,
            1961,
            1514,
            609,
            1421,
            409,
            870,
            1462,
            1863,
            992,
            357,
            200
        ],
        [
            617,
            984,
            233,
            916,
            1865,
            1961,
            1514,
            994,
            33,
            409,
            1015,
            1035,
            725,
            1676,
            1180,
            248
        ],
        [
            617,
            984,
            233,
            916,
            1865,
            1961,
            1514,
            609,
            740,
            409,
            75,
            954,
            1863,
            257,
            357,
            248
        ],
        [
            1733,
            223,
            2019,
            916,
            1728,
            1961,
            100,
            571,
            1599,
            129,
            398,
            1185,
            899,
            5,
            881,
            40
        ],
        [
            1318,
            984,
            233,
            916,
            94,
            1961,
            1455,
            571,
            397,
            811,
            299,
            455,
            954,
            1931,
            10,
            1042
        ],
        [
            1318,
            223,
            42,
            916,
            1728,
            1961,
            830,
            69,
            489,
            899,
            1015,
            1089,
            174,
            1573,
            10,
            178
        ],
        [
            1318,
            984,
            233,
            916,
            299,
            1961,
            395,
            1543,
            532,
            484,
            299,
            106,
            954,
            611,
            10,
            1127
        ],
        [
            1318,
            223,
            42,
            916,
            1728,
            1961,
            830,
            1204,
            773,
            899,
            299,
            1172,
            1077,
            1931,
            10,
            157
        ],
        [
            1318,
            223,
            42,
            916,
            1555,
            1961,
            830,
            983,
            1014,
            91,
            176,
            1089,
            174,
            792,
            10,
            65
        ],
        [
            1318,
            223,
            42,
            916,
            1728,
            1961,
            830,
            1204,
            1014,
            129,
            299,
            857,
            174,
            255,
            10,
            555
        ],
        [
            1318,
            1408,
            1114,
            1173,
            299,
            1961,
            1699,
            1204,
            672,
            129,
            299,
            518,
            1863,
            88,
            726,
            112
        ],
        [
            1318,
            1408,
            42,
            382,
            1535,
            1961,
            1699,
            200,
            489,
            100,
            1015,
            870,
            156,
            259,
            30,
            703
        ],
        [
            1318,
            1408,
            42,
            382,
            1535,
            1961,
            1247,
            200,
            489,
            811,
            1015,
            606,
            584,
            1154,
            1740,
            797
        ],
        [
            1318,
            1408,
            42,
            916,
            242,
            1961,
            1247,
            1958,
            467,
            899,
            1015,
            1841,
            174,
            88,
            24,
            707
        ],
        [
            1318,
            1408,
            42,
            916,
            242,
            1961,
            1247,
            1958,
            467,
            899,
            1015,
            1841,
            205,
            1573,
            335,
            217
        ],
        [
            1318,
            1184,
            215,
            916,
            1764,
            1961,
            100,
            1958,
            13,
            899,
            1015,
            1172,
            174,
            401,
            1856,
            125
        ],
        [
            1318,
            682,
            594,
            916,
            361,
            1961,
            1247,
            1958,
            672,
            899,
            1015,
            1841,
            109,
            397,
            335,
            1182
        ],
        [
            1318,
            682,
            215,
            382,
            1267,
            1961,
            100,
            297,
            532,
            129,
            1446,
            1689,
            1159,
            1324,
            155,
            1156
        ],
        [
            1318,
            682,
            215,
            382,
            1385,
            1961,
            1161,
            1512,
            532,
            1963,
            1015,
            496,
            156,
            119,
            950,
            1156
        ],
        [
            1318,
            682,
            215,
            382,
            1385,
            1961,
            1161,
            1512,
            532,
            1963,
            338,
            559,
            156,
            1154,
            1190,
            1640
        ],
        [
            1449,
            682,
            215,
            1268,
            1385,
            1961,
            1247,
            72,
            9,
            1963,
            869,
            802,
            1863,
            1110,
            428,
            490
        ],
        [
            1318,
            682,
            215,
            1268,
            1385,
            1961,
            1247,
            72,
            1337,
            943,
            972,
            1055,
            969,
            584,
            408,
            1281
        ],
        [
            1318,
            682,
            215,
            1268,
            1385,
            1961,
            1161,
            1964,
            101,
            409,
            418,
            173,
            480,
            1573,
            319,
            1156
        ],
        [
            1318,
            682,
            215,
            1268,
            1385,
            1961,
            1247,
            72,
            1337,
            943,
            418,
            1460,
            104,
            2040,
            155,
            44
        ],
        [
            1318,
            682,
            215,
            1268,
            1385,
            1961,
            1247,
            72,
            1337,
            943,
            418,
            1055,
            104,
            609,
            1086,
            44
        ],
        [
            1318,
            682,
            215,
            1268,
            1385,
            1961,
            1247,
            72,
            1337,
            943,
            418,
            1055,
            104,
            609,
            1086,
            44
        ],
        [
            1318,
            682,
            215,
            382,
            1334,
            1961,
            1247,
            1913,
            489,
            1320,
            1015,
            805,
            1199,
            1324,
            618,
            14
        ],
        [
            1318,
            682,
            215,
            382,
            1334,
            1961,
            1247,
            1369,
            489,
            1320,
            1015,
            606,
            1199,
            27,
            138,
            180
        ],
        [
            1318,
            1408,
            215,
            382,
            1334,
            1961,
            1247,
            1369,
            489,
            1320,
            1015,
            805,
            1199,
            27,
            138,
            180
        ],
        [
            1354,
            1408,
            215,
            382,
            242,
            1961,
            886,
            964,
            532,
            67,
            1015,
            805,
            129,
            1384,
            138,
            144
        ],
        [
            1263,
            396,
            215,
            1436,
            1535,
            791,
            436,
            882,
            1410,
            129,
            672,
            1006,
            346,
            407,
            10,
            1281
        ],
        [
            1263,
            396,
            215,
            1436,
            1535,
            791,
            436,
            882,
            1410,
            129,
            299,
            857,
            174,
            407,
            298,
            120
        ],
        [
            1263,
            396,
            215,
            1310,
            1535,
            1961,
            3,
            1987,
            1444,
            1009,
            1015,
            805,
            1644,
            188,
            1631,
            1027
        ],
        [
            1263,
            1023,
            287,
            1105,
            681,
            1961,
            1523,
            983,
            606,
            1755,
            176,
            384,
            375,
            188,
            535,
            356
        ],
        [
            1263,
            1023,
            287,
            1105,
            681,
            1961,
            1523,
            983,
            606,
            1755,
            176,
            384,
            375,
            188,
            223,
            356
        ],
        [
            1318,
            1227,
            1080,
            294,
            682,
            1961,
            1247,
            297,
            1599,
            129,
            299,
            1177,
            579,
            1111,
            618,
            349
        ],
        [
            1383,
            1227,
            233,
            382,
            1865,
            1961,
            1247,
            1913,
            489,
            409,
            869,
            857,
            347,
            2006,
            165,
            144
        ],
        [
            553,
            1408,
            233,
            382,
            1641,
            1961,
            886,
            33,
            164,
            1145,
            1554,
            1882,
            188,
            329,
            259,
            57
        ],
        [
            553,
            1408,
            42,
            1871,
            890,
            1961,
            988,
            1349,
            41,
            193,
            1884,
            1006,
            1101,
            329,
            731,
            1471
        ],
        [
            1383,
            1408,
            1114,
            1173,
            361,
            1961,
            1247,
            297,
            1599,
            129,
            23,
            455,
            1199,
            340,
            525,
            415
        ],
        [
            1383,
            682,
            594,
            1139,
            681,
            1961,
            1247,
            571,
            489,
            899,
            1015,
            805,
            1199,
            1324,
            266,
            1232
        ],
        [
            1383,
            682,
            509,
            294,
            681,
            1961,
            395,
            88,
            672,
            606,
            11,
            878,
            703,
            297,
            259,
            1095
        ],
        [
            1383,
            682,
            509,
            1759,
            681,
            1961,
            100,
            447,
            1710,
            1320,
            869,
            67,
            1077,
            932,
            382,
            326
        ],
        [
            1383,
            682,
            509,
            1114,
            681,
            1961,
            1247,
            69,
            1156,
            129,
            299,
            857,
            156,
            932,
            1439,
            326
        ],
        [
            1354,
            682,
            215,
            382,
            1334,
            1961,
            1247,
            104,
            532,
            129,
            1015,
            1605,
            156,
            296,
            211,
            552
        ],
        [
            1354,
            682,
            215,
            382,
            1175,
            1961,
            30,
            1811,
            532,
            368,
            609,
            417,
            297,
            1772,
            428,
            1156
        ],
        [
            1734,
            168,
            215,
            382,
            1334,
            1961,
            1523,
            979,
            532,
            899,
            609,
            1578,
            455,
            1027,
            1279,
            1263
        ],
        [
            617,
            1720,
            1114,
            1114,
            322,
            1961,
            1247,
            297,
            431,
            810,
            1828,
            106,
            547,
            245,
            476,
            18
        ],
        [
            617,
            1720,
            1114,
            1114,
            322,
            1961,
            1247,
            297,
            431,
            810,
            441,
            357,
            1159,
            644,
            441,
            490
        ],
        [
            1720,
            682,
            215,
            382,
            1334,
            1961,
            100,
            1275,
            1156,
            1913,
            1015,
            106,
            129,
            391,
            1726,
            381
        ]
    ]
        )
        EXPECTED_CODES_1 = torch.tensor(
            [
        [
            391,
            855,
            73,
            382,
            783,
            980,
            590,
            609,
            714,
            25,
            982,
            1185,
            16,
            97,
            615,
            767
        ],
        [
            415,
            937,
            651,
            382,
            648,
            267,
            696,
            609,
            3,
            1788,
            1720,
            306,
            360,
            241,
            151,
            480
        ],
        [
            1029,
            1045,
            1425,
            1448,
            121,
            267,
            1087,
            536,
            826,
            810,
            550,
            1624,
            1146,
            765,
            808,
            861
        ],
        [
            1029,
            590,
            1090,
            158,
            423,
            1185,
            1967,
            1913,
            1067,
            116,
            297,
            55,
            2041,
            55,
            722,
            368
        ],
        [
            1029,
            348,
            1730,
            160,
            1684,
            1533,
            1706,
            1986,
            830,
            463,
            593,
            214,
            176,
            1668,
            182,
            1200
        ],
        [
            695,
            348,
            508,
            27,
            426,
            176,
            389,
            1770,
            1592,
            429,
            1958,
            1435,
            1158,
            277,
            920,
            555
        ],
        [
            695,
            590,
            336,
            565,
            1153,
            911,
            572,
            233,
            1202,
            1795,
            1379,
            702,
            1038,
            673,
            1354,
            918
        ],
        [
            695,
            590,
            1052,
            981,
            254,
            0,
            1362,
            37,
            1879,
            961,
            712,
            67,
            1411,
            107,
            460,
            882
        ],
        [
            687,
            590,
            1052,
            981,
            254,
            1287,
            777,
            492,
            293,
            196,
            276,
            1985,
            1936,
            733,
            458,
            22
        ],
        [
            1354,
            693,
            634,
            491,
            131,
            1011,
            331,
            143,
            164,
            1168,
            1959,
            100,
            1707,
            1797,
            282,
            827
        ],
        [
            1639,
            844,
            679,
            962,
            211,
            1082,
            1374,
            1813,
            924,
            561,
            322,
            672,
            128,
            718,
            1484,
            350
        ],
        [
            1639,
            614,
            1666,
            5,
            317,
            587,
            1747,
            974,
            420,
            200,
            262,
            909,
            85,
            109,
            133,
            187
        ],
        [
            302,
            1956,
            1666,
            962,
            197,
            200,
            793,
            1223,
            1672,
            644,
            689,
            1065,
            8,
            24,
            111,
            800
        ],
        [
            1570,
            847,
            1666,
            5,
            197,
            146,
            63,
            711,
            149,
            161,
            1755,
            1413,
            134,
            266,
            630,
            837
        ],
        [
            1570,
            614,
            1666,
            420,
            1994,
            285,
            1363,
            185,
            280,
            387,
            686,
            316,
            1101,
            619,
            177,
            1742
        ],
        [
            1570,
            614,
            1666,
            452,
            1998,
            721,
            367,
            711,
            336,
            850,
            283,
            156,
            355,
            806,
            1171,
            1045
        ],
        [
            1570,
            693,
            1666,
            452,
            964,
            721,
            1363,
            617,
            108,
            850,
            250,
            20,
            1328,
            298,
            1130,
            415
        ],
        [
            1570,
            1875,
            787,
            563,
            478,
            321,
            1431,
            424,
            654,
            636,
            1088,
            599,
            908,
            806,
            956,
            88
        ],
        [
            1570,
            781,
            787,
            745,
            1525,
            586,
            962,
            52,
            454,
            1282,
            518,
            85,
            347,
            1701,
            110,
            480
        ],
        [
            439,
            693,
            938,
            1753,
            1671,
            214,
            915,
            2029,
            1242,
            1124,
            223,
            175,
            783,
            344,
            85,
            1610
        ],
        [
            439,
            787,
            1054,
            750,
            1632,
            879,
            1751,
            1495,
            101,
            1782,
            242,
            1357,
            899,
            586,
            404,
            679
        ],
        [
            439,
            781,
            1305,
            638,
            1120,
            585,
            1103,
            293,
            101,
            505,
            301,
            728,
            174,
            662,
            708,
            1335
        ],
        [
            439,
            137,
            482,
            1290,
            247,
            601,
            762,
            1333,
            1409,
            1133,
            412,
            38,
            174,
            1218,
            1466,
            67
        ],
        [
            1354,
            1838,
            2007,
            111,
            1622,
            85,
            63,
            1333,
            716,
            1596,
            217,
            38,
            70,
            417,
            1221,
            2
        ],
        [
            1354,
            693,
            1635,
            1871,
            1865,
            1689,
            63,
            1,
            919,
            961,
            249,
            888,
            609,
            1871,
            1520,
            124
        ],
        [
            1354,
            614,
            1666,
            1253,
            1316,
            587,
            1258,
            740,
            279,
            1475,
            950,
            448,
            715,
            38,
            1259,
            965
        ],
        [
            1354,
            614,
            1666,
            1253,
            1970,
            587,
            367,
            1813,
            70,
            216,
            1063,
            156,
            3,
            110,
            67,
            623
        ],
        [
            302,
            1132,
            1231,
            745,
            1358,
            1657,
            862,
            471,
            235,
            1009,
            176,
            1015,
            1508,
            649,
            1570,
            1072
        ],
        [
            302,
            1149,
            1162,
            1315,
            476,
            1239,
            95,
            471,
            208,
            69,
            282,
            104,
            59,
            434,
            1374,
            1995
        ],
        [
            302,
            324,
            1231,
            1542,
            638,
            505,
            1212,
            302,
            725,
            180,
            938,
            303,
            234,
            258,
            202,
            574
        ],
        [
            302,
            1149,
            482,
            382,
            1449,
            700,
            100,
            439,
            1849,
            304,
            1472,
            38,
            29,
            106,
            53,
            41
        ],
        [
            1306,
            1149,
            482,
            2007,
            1905,
            700,
            193,
            1017,
            208,
            1587,
            176,
            1736,
            1682,
            258,
            1515,
            276
        ],
        [
            1306,
            1149,
            482,
            610,
            1201,
            88,
            117,
            44,
            208,
            2000,
            505,
            618,
            970,
            434,
            568,
            117
        ],
        [
            1306,
            284,
            940,
            632,
            1938,
            326,
            1759,
            2031,
            810,
            86,
            1042,
            1047,
            706,
            2027,
            82,
            1435
        ],
        [
            1100,
            1722,
            2025,
            241,
            149,
            121,
            1300,
            405,
            564,
            1786,
            743,
            823,
            783,
            1759,
            812,
            803
        ],
        [
            1954,
            1722,
            355,
            864,
            914,
            819,
            944,
            405,
            1728,
            1144,
            679,
            903,
            783,
            1529,
            1008,
            803
        ],
        [
            1543,
            1722,
            355,
            864,
            914,
            819,
            944,
            278,
            1728,
            1673,
            2040,
            903,
            996,
            1838,
            1008,
            901
        ],
        [
            1841,
            1722,
            355,
            371,
            1296,
            1093,
            625,
            1814,
            930,
            1144,
            679,
            362,
            1110,
            1838,
            1772,
            1668
        ],
        [
            1841,
            1722,
            355,
            371,
            1296,
            1093,
            625,
            1814,
            930,
            1144,
            679,
            362,
            1110,
            1838,
            1772,
            1668
        ],
        [
            898,
            1722,
            355,
            371,
            1296,
            1093,
            625,
            1814,
            930,
            1144,
            679,
            362,
            1110,
            1838,
            1772,
            1668
        ],
        [
            859,
            1722,
            355,
            371,
            1296,
            1093,
            625,
            1814,
            930,
            1144,
            679,
            1247,
            1110,
            889,
            1772,
            803
        ],
        [
            1710,
            858,
            580,
            1594,
            429,
            343,
            222,
            133,
            1842,
            551,
            311,
            1656,
            13,
            241,
            2034,
            1116
        ],
        [
            1311,
            1856,
            1691,
            981,
            1727,
            35,
            1627,
            114,
            2043,
            1777,
            114,
            492,
            164,
            465,
            41,
            1875
        ],
        [
            866,
            1856,
            82,
            458,
            1183,
            563,
            1627,
            963,
            540,
            221,
            1776,
            1,
            74,
            1229,
            129,
            1024
        ],
        [
            866,
            1010,
            1305,
            528,
            840,
            1143,
            969,
            1958,
            892,
            221,
            707,
            234,
            1505,
            269,
            53,
            752
        ],
        [
            855,
            1010,
            750,
            1824,
            1226,
            1961,
            1680,
            40,
            628,
            783,
            25,
            702,
            1505,
            113,
            149,
            663
        ],
        [
            855,
            1569,
            1305,
            1097,
            1226,
            1961,
            603,
            1333,
            628,
            81,
            158,
            1073,
            139,
            705,
            1130,
            263
        ],
        [
            63,
            1569,
            1305,
            1097,
            1226,
            1961,
            603,
            1333,
            286,
            102,
            220,
            1073,
            1789,
            6,
            277,
            263
        ],
        [
            808,
            614,
            679,
            69,
            374,
            587,
            268,
            110,
            2037,
            258,
            1469,
            1250,
            1965,
            998,
            266,
            88
        ],
        [
            1570,
            1838,
            2007,
            158,
            681,
            573,
            862,
            1534,
            955,
            399,
            870,
            165,
            492,
            1445,
            366,
            194
        ],
        [
            1570,
            1149,
            756,
            1520,
            700,
            435,
            742,
            55,
            806,
            1908,
            505,
            98,
            1234,
            76,
            662,
            728
        ],
        [
            302,
            1838,
            1234,
            1595,
            917,
            49,
            100,
            750,
            51,
            1009,
            261,
            857,
            37,
            302,
            492,
            166
        ],
        [
            302,
            1923,
            1364,
            1367,
            392,
            227,
            100,
            378,
            100,
            291,
            176,
            38,
            1994,
            106,
            162,
            210
        ],
        [
            302,
            1923,
            908,
            487,
            1824,
            176,
            60,
            1244,
            208,
            71,
            299,
            264,
            384,
            1326,
            298,
            1007
        ],
        [
            302,
            1873,
            1364,
            1197,
            462,
            109,
            100,
            378,
            208,
            71,
            403,
            264,
            303,
            434,
            298,
            322
        ],
        [
            302,
            1873,
            1364,
            1197,
            462,
            109,
            100,
            360,
            100,
            71,
            16,
            420,
            303,
            434,
            298,
            322
        ],
        [
            302,
            1512,
            1364,
            1197,
            462,
            1753,
            100,
            633,
            1009,
            1768,
            146,
            420,
            303,
            131,
            492,
            234
        ],
        [
            302,
            1512,
            1364,
            1197,
            462,
            1753,
            678,
            325,
            780,
            71,
            403,
            420,
            183,
            188,
            219,
            322
        ],
        [
            302,
            273,
            1364,
            1197,
            462,
            1753,
            678,
            633,
            548,
            461,
            247,
            420,
            303,
            629,
            492,
            892
        ],
        [
            302,
            273,
            1364,
            1197,
            462,
            1753,
            678,
            633,
            548,
            461,
            247,
            420,
            623,
            1078,
            492,
            370
        ],
        [
            302,
            273,
            1364,
            1197,
            462,
            1753,
            678,
            633,
            548,
            461,
            247,
            420,
            623,
            1078,
            492,
            370
        ],
        [
            1570,
            1615,
            568,
            1612,
            462,
            1407,
            177,
            653,
            208,
            317,
            403,
            70,
            55,
            196,
            397,
            322
        ],
        [
            1570,
            1293,
            1666,
            641,
            250,
            86,
            1258,
            711,
            286,
            1745,
            620,
            20,
            485,
            1203,
            749,
            837
        ],
        [
            1570,
            62,
            1231,
            1110,
            462,
            109,
            1250,
            346,
            1106,
            68,
            25,
            420,
            70,
            199,
            11,
            1555
        ],
        [
            1570,
            1293,
            1231,
            583,
            462,
            1650,
            358,
            708,
            1106,
            180,
            151,
            982,
            16,
            434,
            1478,
            166
        ],
        [
            1570,
            1615,
            568,
            1612,
            462,
            762,
            177,
            653,
            208,
            1939,
            747,
            1606,
            143,
            434,
            15,
            1027
        ],
        [
            1570,
            1293,
            1231,
            583,
            462,
            1650,
            1078,
            708,
            208,
            71,
            204,
            70,
            1300,
            434,
            1478,
            90
        ],
        [
            1570,
            1293,
            1231,
            1675,
            462,
            348,
            1949,
            633,
            209,
            96,
            505,
            1051,
            65,
            434,
            162,
            1065
        ],
        [
            1570,
            1149,
            1244,
            1675,
            1651,
            821,
            358,
            302,
            208,
            180,
            786,
            319,
            229,
            1018,
            662,
            555
        ],
        [
            1892,
            1293,
            1231,
            1675,
            462,
            1340,
            1250,
            346,
            662,
            96,
            803,
            1051,
            508,
            856,
            11,
            2042
        ],
        [
            1892,
            1293,
            1094,
            1675,
            462,
            1407,
            1096,
            633,
            167,
            96,
            803,
            223,
            124,
            196,
            11,
            1767
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            803,
            223,
            124,
            9,
            562,
            555
        ],
        [
            1892,
            1293,
            1094,
            1834,
            462,
            1751,
            932,
            1829,
            19,
            122,
            559,
            1973,
            59,
            1055,
            298,
            113
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            535,
            124,
            1726,
            309,
            214
        ],
        [
            1892,
            1293,
            1094,
            1020,
            462,
            1340,
            1250,
            633,
            446,
            1587,
            559,
            223,
            827,
            188,
            10,
            88
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            161
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            161
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            161
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            88
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            161
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            161
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            494,
            803,
            223,
            124,
            1726,
            309,
            555
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            407,
            480,
            309,
            88
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            326,
            1587,
            559,
            223,
            827,
            1726,
            309,
            200
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            197,
            1587,
            559,
            223,
            827,
            1726,
            911,
            200
        ],
        [
            1892,
            1293,
            1037,
            87,
            462,
            739,
            358,
            325,
            208,
            1932,
            173,
            927,
            75,
            768,
            509,
            200
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            1250,
            633,
            197,
            1587,
            559,
            223,
            424,
            1726,
            11,
            88
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            494,
            803,
            223,
            124,
            1726,
            309,
            555
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            1587,
            803,
            223,
            124,
            1726,
            562,
            555
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            1587,
            605,
            535,
            124,
            1726,
            11,
            200
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            494,
            803,
            223,
            124,
            196,
            11,
            200
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            1587,
            803,
            223,
            124,
            1726,
            562,
            555
        ],
        [
            1892,
            1293,
            1094,
            1315,
            462,
            1340,
            212,
            633,
            197,
            494,
            803,
            223,
            124,
            196,
            11,
            200
        ],
        [
            439,
            614,
            1666,
            69,
            1377,
            151,
            1240,
            2024,
            448,
            457,
            1348,
            1026,
            1570,
            611,
            1040,
            1350
        ],
        [
            439,
            614,
            1666,
            69,
            362,
            151,
            1039,
            74,
            245,
            672,
            1680,
            1336,
            334,
            1428,
            1040,
            1851
        ],
        [
            1224,
            614,
            1666,
            676,
            441,
            1008,
            11,
            27,
            1241,
            385,
            128,
            391,
            842,
            16,
            354,
            974
        ],
        [
            1224,
            614,
            1666,
            676,
            441,
            1008,
            63,
            27,
            1119,
            899,
            128,
            984,
            1193,
            961,
            1171,
            121
        ],
        [
            1224,
            614,
            1666,
            676,
            441,
            1008,
            63,
            27,
            1119,
            899,
            128,
            984,
            1295,
            10,
            1171,
            568
        ],
        [
            658,
            582,
            1638,
            2033,
            644,
            198,
            930,
            521,
            1881,
            235,
            127,
            169,
            1239,
            804,
            584,
            940
        ]
    ]
        )
        # fmt: on
        torch.testing.assert_close(codes_list[0].cpu(), EXPECTED_CODES_0)
        torch.testing.assert_close(codes_list[1].cpu(), EXPECTED_CODES_1)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_with_speaker(self):
        """TTS with a named speaker (requires model to expose speaker list)."""
        model, processor = self._load_model_and_processor()

        supported_speakers = model.get_supported_speakers()
        if not supported_speakers:
            self.skipTest("Model has no built-in speakers; skipping speaker test.")

        speaker = supported_speakers[0]
        text = "Hello from your favourite speaker."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, _ = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            speakers=[speaker],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=100,
        )

        self.assertEqual(len(codes_list), 1)
        codes = codes_list[0]
        self.assertEqual(codes.dim(), 2)
        self.assertEqual(codes.shape[-1], model.talker.config.num_code_groups)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_sampling(self):
        """Stochastic generation (do_sample=True) produces codes of the expected shape."""
        model, processor = self._load_model_and_processor()

        text = "Hello, this is a stochastic generation test."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, hidden_list = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=1.0,
            subtalker_dosample=True,
            max_new_tokens=50,
        )

        self.assertEqual(len(codes_list), 1)
        codes = codes_list[0]
        self.assertEqual(codes.dim(), 2)
        self.assertEqual(codes.shape[-1], model.talker.config.num_code_groups)
        self.assertEqual(len(hidden_list), 1)
        self.assertEqual(hidden_list[0].shape[0], codes.shape[0])

    @slow
    @require_torch_accelerator
    def test_small_model_integration_max_new_tokens_respected(self):
        """max_new_tokens limits the length of the generated code sequence."""
        model, processor = self._load_model_and_processor()

        text = "This sentence should be truncated early by the max_new_tokens limit."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        limit = 20
        codes_list, _ = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=limit,
        )

        self.assertEqual(len(codes_list), 1)
        self.assertLessEqual(codes_list[0].shape[0], limit)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_output_types(self):
        """generate() returns lists of CPU-movable tensors with correct dtypes."""
        model, processor = self._load_model_and_processor()

        text = "Output type verification."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, hidden_list = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=30,
        )

        self.assertIsInstance(codes_list, list)
        self.assertIsInstance(hidden_list, list)
        codes = codes_list[0]
        # Codes must be integer tensors (codec token indices)
        self.assertTrue(codes.dtype in (torch.int32, torch.int64, torch.long))
        # Hidden states must be floating-point
        self.assertTrue(hidden_list[0].dtype in (torch.float16, torch.bfloat16, torch.float32))

    @slow
    @require_torch_accelerator
    def test_small_model_integration_hidden_states_shape(self):
        """Hidden states returned alongside codes have a consistent sequence length."""
        model, processor = self._load_model_and_processor()

        text = "Verify hidden state shapes match code sequence length."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, hidden_list = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=50,
        )

        codes = codes_list[0]
        hidden = hidden_list[0]
        # First dimension of hidden states must equal the number of generated code frames
        self.assertEqual(hidden.shape[0], codes.shape[0])
        # Hidden size must be positive
        self.assertGreater(hidden.shape[-1], 0)

    @slow
    @require_torch_accelerator
    def test_small_model_integration_batch_independent(self):
        """Each item in a batch generates independently reproducible codes."""
        model, processor = self._load_model_and_processor()

        text = "The weather is nice today."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        # Generate single item
        codes_single, _ = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=50,
        )
        # Generate same item twice in a batch
        codes_batch, _ = model.generate(
            input_ids=[inputs["input_ids"], inputs["input_ids"]],
            languages=["auto", "auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=50,
        )

        self.assertEqual(len(codes_batch), 2)
        # Two identical inputs in the same batch must produce identical codes —
        # verifying there is no cross-item contamination in the batch generation.
        torch.testing.assert_close(codes_batch[0], codes_batch[1])

    @slow
    @require_torch_accelerator
    def test_small_model_integration_text_to_audio_v2(self):
        """End-to-end: text → codes → audio waveform using the V2 (12Hz) speech tokenizer."""
        model, processor = self._load_model_and_processor()

        text = "Hello, this is an end-to-end test."
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        codes_list, _ = model.generate(
            input_ids=[inputs["input_ids"]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=50,
        )
        self.assertEqual(len(codes_list), 1)
        codes = codes_list[0]

        # Load V2 speech tokenizer and decode codes → audio
        speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", subfolder="speech_tokenizer", device_map=torch_device
        )
        speech_tokenizer.eval()

        # codes shape from generate: (seq_len, num_code_groups)
        # V2 decode expects: (batch, codes_len, num_quantizers)
        audio_codes = codes.unsqueeze(0).to(torch_device)
        with torch.no_grad():
            output = speech_tokenizer.decode(audio_codes, return_dict=True)

        self.assertEqual(len(output.audio_values), 1)
        wav = output.audio_values[0]

        # Validate audio output
        self.assertTrue(wav.is_floating_point(), "Waveform should be float")
        self.assertGreater(len(wav), 0, "Waveform should not be empty")
        self.assertTrue(torch.isfinite(wav).all(), "Waveform should contain no inf values")
        self.assertFalse(torch.isnan(wav).any(), "Waveform should contain no NaN values")
        self.assertLessEqual(wav.max().item(), 1.0, "Waveform should be clamped to [-1, 1]")
        self.assertGreaterEqual(wav.min().item(), -1.0, "Waveform should be clamped to [-1, 1]")

        # Audio duration should be proportional to codes length.
        # chunked_decode may produce slightly fewer samples than the exact formula
        # due to chunk boundary effects, so allow a small tolerance.
        expected_audio_length = codes.shape[0] * speech_tokenizer.decode_upsample_rate
        self.assertGreater(len(wav), 0)
        self.assertLessEqual(len(wav), expected_audio_length)
        # Must be at least 95% of the expected length
        self.assertGreater(len(wav), int(expected_audio_length * 0.95))

    @slow
    @require_torch_accelerator
    def test_small_model_integration_batch_text_to_audio_v2(self):
        """End-to-end batch: multiple texts → codes → audio waveforms."""
        model, processor = self._load_model_and_processor()

        texts = ["Hello.", "The weather is nice today."]
        inputs_list = [
            processor(text=t, return_tensors="pt").to(torch_device)
            for t in texts
        ]

        codes_list, _ = model.generate(
            input_ids=[inp["input_ids"] for inp in inputs_list],
            languages=["auto", "auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=50,
        )
        self.assertEqual(len(codes_list), 2)

        # Load V2 speech tokenizer
        speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", subfolder="speech_tokenizer", device_map=torch_device
        )
        speech_tokenizer.eval()

        # Decode each item independently (variable lengths)
        for i, codes in enumerate(codes_list):
            audio_codes = codes.unsqueeze(0).to(torch_device)
            with torch.no_grad():
                output = speech_tokenizer.decode(audio_codes, return_dict=True)
            wav = output.audio_values[0]
            self.assertTrue(wav.is_floating_point())
            self.assertGreater(len(wav), 0, f"Waveform {i} should not be empty")
            self.assertFalse(torch.isnan(wav).any(), f"Waveform {i} has NaNs")
            self.assertTrue(torch.isfinite(wav).all(), f"Waveform {i} has infs")

    @slow
    @require_torch_accelerator
    def test_small_model_integration_v2_tokenizer_encode_decode_roundtrip(self):
        """V2 tokenizer encode → decode roundtrip preserves audio structure (real weights)."""
        speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", subfolder="speech_tokenizer", device_map=torch_device
        )
        speech_tokenizer.eval()

        # Create a synthetic 24kHz audio signal (1 second sine wave)
        sample_rate = speech_tokenizer.input_sample_rate
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration), device=torch_device)
        audio = 0.5 * torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz tone

        # Encode: audio → codes
        input_values = audio.unsqueeze(0)
        padding_mask = torch.ones_like(input_values, dtype=torch.long)
        with torch.no_grad():
            encoded = speech_tokenizer.encode(input_values, padding_mask=padding_mask, return_dict=True)
        self.assertEqual(len(encoded.audio_codes), 1)
        codes = encoded.audio_codes[0]  # (codes_len, num_quantizers)
        self.assertEqual(codes.dim(), 2)
        self.assertEqual(codes.shape[-1], speech_tokenizer.encoder_valid_num_quantizers)

        # Decode: codes → audio
        with torch.no_grad():
            decoded = speech_tokenizer.decode(codes.unsqueeze(0), return_dict=True)
        wav = decoded.audio_values[0]
        self.assertTrue(wav.is_floating_point())
        self.assertGreater(len(wav), 0)
        self.assertFalse(torch.isnan(wav).any())

    @slow
    @require_torch_accelerator
    def test_small_model_integration_v2_audio_duration_proportional_to_codes(self):
        """V2 tokenizer: audio duration doubles when code sequence doubles (real weights)."""
        speech_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base", subfolder="speech_tokenizer", device_map=torch_device
        )
        speech_tokenizer.eval()

        num_q = speech_tokenizer.encoder_valid_num_quantizers
        codebook_size = speech_tokenizer.decoder.config.codebook_size

        codes_short = torch.randint(1, codebook_size, (1, 10, num_q), device=torch_device)
        codes_long = torch.randint(1, codebook_size, (1, 20, num_q), device=torch_device)

        with torch.no_grad():
            wav_short = speech_tokenizer.decode(codes_short, return_dict=True).audio_values[0]
            wav_long = speech_tokenizer.decode(codes_long, return_dict=True).audio_values[0]

        # Longer codes must produce strictly longer audio.
        # Exact 2x ratio doesn't hold due to chunked_decode boundary effects,
        # but the ratio must be between 1.8x and 2.2x of the shorter audio.
        self.assertGreater(len(wav_long), len(wav_short))
        ratio = len(wav_long) / len(wav_short)
        self.assertGreater(ratio, 1.8)
        self.assertLess(ratio, 2.2)
