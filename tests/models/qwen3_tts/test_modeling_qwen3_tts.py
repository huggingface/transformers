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
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor

if is_torch_available():
    import torch
    from transformers import (
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
        Qwen3TTSTalkerConfig,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTokenizerV1Config,
        Qwen3TTSTokenizerV2Config,
    )
    from transformers.models.qwen3_tts.modeling_qwen3_tts import (
        Qwen3TTSTokenizerV1Decoder,
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
    def test_small_model_integration_text_to_codes(self):
        """Text -> acoustic codec codes; checks output shape and token range."""
        model, processor = self._load_model_and_processor()

        text = "Hello, how are you doing today?"
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        # generate() expects a list of 1-D / 2-D token tensors
        codes_list, _ = model.generate(
            input_ids=[inputs["input_ids"][0]],
            languages=["auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=100,
        )

        # One output sequence per input
        self.assertEqual(len(codes_list), 1)

        codes = codes_list[0]  # (T, num_code_groups)
        num_code_groups = model.talker.config.num_code_groups
        vocab_size = model.talker.config.vocab_size

        self.assertEqual(codes.dim(), 2)
        self.assertEqual(codes.shape[-1], num_code_groups)
        # All code tokens must be in [0, vocab_size)
        self.assertTrue(codes.ge(0).all() and codes.lt(vocab_size).all())
        # Must have generated at least a few frames
        self.assertGreater(codes.shape[0], 1)

        # fmt: off
        # NOTE: Update these expected values by running:
        #   codes_list, _ = model.generate(input_ids=[inputs["input_ids"][0]], languages=["auto"],
        #                                  do_sample=False, subtalker_dosample=False, max_new_tokens=100)
        #   print(codes_list[0][:5].tolist())
        EXPECTED_FIRST_5_FRAMES = None  # Replace with actual values once weights are available
        # fmt: on
        if EXPECTED_FIRST_5_FRAMES is not None:
            torch.testing.assert_close(codes[:5].cpu(), torch.tensor(EXPECTED_FIRST_5_FRAMES))

    @slow
    def test_small_model_integration_batch(self):
        """Batch: two texts → two independent code sequences."""
        model, processor = self._load_model_and_processor()

        texts = [
            "The weather is nice today.",
            "I enjoy listening to music.",
        ]
        inputs_0 = processor(text=texts[0], return_tensors="pt").to(torch_device)
        inputs_1 = processor(text=texts[1], return_tensors="pt").to(torch_device)

        codes_list, _ = model.generate(
            input_ids=[inputs_0["input_ids"][0], inputs_1["input_ids"][0]],
            languages=["auto", "auto"],
            do_sample=False,
            subtalker_dosample=False,
            max_new_tokens=100,
        )

        self.assertEqual(len(codes_list), 2)
        num_code_groups = model.talker.config.num_code_groups

        for codes in codes_list:
            self.assertEqual(codes.dim(), 2)
            self.assertEqual(codes.shape[-1], num_code_groups)
            self.assertGreater(codes.shape[0], 1)

    @slow
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
            input_ids=[inputs["input_ids"][0]],
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
