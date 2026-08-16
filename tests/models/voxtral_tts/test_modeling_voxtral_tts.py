# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Voxtral TTS model."""

import copy
import os
import shutil
import tempfile
import unittest

import pytest

from transformers import VoxtralTtsConfig, is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    ids_tensor,
)


if is_torch_available():
    import torch

    from transformers import VoxtralTtsForTextToSpeech
    from transformers.models.voxtral_tts.convert_voxtral_tts_weights_to_hf import write_model


class VoxtralTtsModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        num_audio_frames=3,
        is_training=False,
        num_codebooks=37,
        hidden_size=64,
        config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "hidden_act": "silu",
            "max_position_embeddings": 64,
            "audio_vocab_size": 256,
            "semantic_codebook_size": 64,
            "acoustic_codebook_size": 5,
            "n_acoustic_codebook": 36,
            "num_codebooks": 37,
            "tie_word_embeddings": True,
        },
        flow_matching_config={
            "input_dim": 64,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "semantic_vocab_size": 64,
            "acoustic_dim": 36,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
            "sigma": 1e-5,
            "sigma_max": 1.0,
        },
        codec_config={
            "semantic_codebook_size": 64,
            "semantic_dim": 32,
            "acoustic_codebook_size": 5,
            "acoustic_dim": 36,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "patch_size": 8,
            "patch_proj_kernel_size": 3,
            "decoder_transformer_lengths": [1, 1, 1, 1],
            "decoder_convs_kernels": [3, 4, 4, 4],
            "decoder_convs_strides": [1, 2, 2, 2],
            "norm_eps": 0.01,
            "qk_norm_eps": 1e-6,
            "layer_scale_init": 0.01,
        },
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_audio_frames = num_audio_frames
        self.is_training = is_training
        self.num_codebooks = num_codebooks
        self.config_dict = config
        self.flow_matching_config = flow_matching_config
        self.codec_config = codec_config

        self.num_hidden_layers = config["num_hidden_layers"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.encoder_seq_length = self.num_audio_frames + self.seq_length

    def get_config(self):
        return VoxtralTtsConfig(
            flow_matching_config=self.flow_matching_config,
            codec_config=self.codec_config,
            **self.config_dict,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)

        audio_codes = torch.stack(
            [
                ids_tensor([self.batch_size, self.num_audio_frames], config.semantic_codebook_size),
            ]
            + [
                ids_tensor([self.batch_size, self.num_audio_frames], config.acoustic_codebook_size)
                for _ in range(config.n_acoustic_codebook)
            ],
            dim=-1,
        )

        attention_mask = torch.ones(
            self.batch_size, self.num_audio_frames + self.seq_length, dtype=torch.long, device=torch_device
        )

        return config, input_ids, audio_codes, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, audio_codes, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "audio_codes": audio_codes,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_forward(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids=input_ids, audio_codes=audio_codes, attention_mask=attention_mask)
        expected_seq_len = self.num_audio_frames + self.seq_length
        self.parent.assertEqual(result.logits.shape, (self.batch_size, expected_seq_len, config.vocab_size))

    def create_and_check_forward_text_only(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(input_ids=input_ids)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))

    def create_and_check_forward_audio_only(self, config, input_ids, audio_codes, attention_mask):
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        result = model(audio_codes=audio_codes)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_audio_frames, config.vocab_size))


@require_torch
class VoxtralTtsModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (VoxtralTtsForTextToSpeech,) if is_torch_available() else ()
    all_generative_model_classes = ()
    test_resize_embeddings = False
    test_resize_embeddings_untied = False

    def setUp(self):
        self.model_tester = VoxtralTtsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralTtsConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (
                    self.model_tester.batch_size,
                    self.model_tester.num_audio_frames + self.model_tester.seq_length,
                ),
                dtype=torch.long,
                device=torch_device,
            )
        return inputs_dict

    def test_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward(*config_and_inputs)

    def test_forward_text_only(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_text_only(*config_and_inputs)

    def test_forward_audio_only(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_forward_audio_only(*config_and_inputs)

    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True

    @pytest.mark.skip(reason="Voxtral TTS has custom embedding approach (text and audio embeddings).")
    def test_model_get_set_embeddings(self):
        pass

    @pytest.mark.skip(reason="Voxtral TTS has no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    @pytest.mark.skip(reason="Voxtral TTS has special text+audio embeddings that are never tied in the standard way.")
    def test_tied_weights_keys(self):
        pass

    @pytest.mark.skip(
        reason="Voxtral TTS uses custom codec/FM modules with layer_scale params and VQ codebook buffers "
        "that require custom _init_weights. The modular converter currently cannot add _init_weights "
        "when the parent (Mistral) doesn't define one."
    )
    def test_can_init_all_missing_weights(self):
        pass

    @pytest.mark.skip(reason="Same as test_can_init_all_missing_weights: codec codebook buffers not in _init_weights.")
    def test_init_weights_can_init_buffers(self):
        pass

    @pytest.mark.skip(
        reason="VoxtralTtsCodecConfig uses strict dataclass validation and layer_scale field type "
        "conflicts with the test's type coercion."
    )
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @pytest.mark.skip(
        reason="inputs_embeds test passes only text-shaped embeds but prepare_config_and_inputs "
        "produces audio+text inputs. The two paths produce different sequence lengths."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @pytest.mark.skip(
        reason="TP plan paths use backbone_model.layers.* but the test resolves "
        "base_model_prefix='model' to backbone_model.backbone_model.layers.*. "
        "This is a known modular converter issue with composite models."
    )
    def test_tp_plan_matches_params(self):
        pass

    def test_generate_with_tiny_model(self):
        """Test that the custom TTS generate method works with a tiny config."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2)
        self.assertIsNotNone(output.audio)
        self.assertEqual(len(output.audio), 1)
        self.assertEqual(output.semantic_tokens.shape[0], 1)
        self.assertEqual(output.semantic_tokens.shape[1], 2)
        self.assertEqual(output.acoustic_values.shape, (1, 2, 36))

    def test_generate_without_audio_output(self):
        """Test generate with output_audio=False returns tokens but no waveform."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2, output_audio=False)
        self.assertIsNone(output.audio)
        self.assertIsNotNone(output.semantic_tokens)
        self.assertIsNotNone(output.acoustic_values)

    def test_generate_with_voice_reference(self):
        """Test generate with audio_codes (voice reference)."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        audio_codes = torch.stack(
            [ids_tensor([1, 3], config.semantic_codebook_size)]
            + [ids_tensor([1, 3], config.acoustic_codebook_size) for _ in range(config.n_acoustic_codebook)],
            dim=-1,
        ).to(torch_device)

        output = model.generate(input_ids=input_ids, audio_codes=audio_codes, max_new_tokens=2)
        self.assertIsNotNone(output.audio)
        self.assertEqual(output.semantic_tokens.shape[1], 2)

    def test_generate_with_cfg(self):
        """Test generate with classifier-free guidance."""
        config = self.model_tester.get_config()
        model = VoxtralTtsForTextToSpeech(config=config)
        model.to(torch_device)
        model.eval()

        input_ids = ids_tensor([1, 5], config.vocab_size).to(torch_device)

        output = model.generate(input_ids=input_ids, max_new_tokens=2, cfg_alpha=1.2)
        self.assertIsNotNone(output.audio)
        self.assertEqual(output.semantic_tokens.shape[1], 2)


@require_torch_accelerator
class VoxtralTtsIntegrationTest(unittest.TestCase):
    model_checkpoint = "mistralai/Voxtral-4B-TTS-2603"
    converted_checkpoint_dir = None
    _cleanup_converted_checkpoint = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.converted_checkpoint_dir = os.environ.get("VOXTRAL_TTS_HF_MODEL_DIR")
        if cls.converted_checkpoint_dir is None:
            cls.converted_checkpoint_dir = tempfile.mkdtemp(prefix="voxtral-tts-hf-")
            cls._cleanup_converted_checkpoint = True
            write_model(cls.model_checkpoint, cls.converted_checkpoint_dir)

    @classmethod
    def tearDownClass(cls):
        if cls._cleanup_converted_checkpoint and cls.converted_checkpoint_dir is not None:
            shutil.rmtree(cls.converted_checkpoint_dir, ignore_errors=True)
        super().tearDownClass()

    def _get_model(self):
        model = VoxtralTtsForTextToSpeech.from_pretrained(
            self.converted_checkpoint_dir, device_map=torch_device, torch_dtype=torch.float16
        )
        model.eval()
        return model

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_forward_logits(self):
        """Test that forward pass matches a known logits slice from converted original weights."""
        model = self._get_model()

        input_ids = torch.tensor([[1, 306, 4658, 278, 6593, 310]], device=torch_device)
        with torch.no_grad():
            out = model(input_ids=input_ids)

        self.assertEqual(out.logits.shape[0], 1)
        self.assertEqual(out.logits.shape[1], 6)
        self.assertEqual(out.logits.shape[2], 131072)
        expected_mean = torch.tensor([[0.0539, 0.1095, 0.0874, 0.1063, 0.0758, 0.1047]])
        expected_last_logits = torch.tensor(
            [
                0.9287,
                -0.0992,
                -2.8809,
                1.0098,
                0.4404,
                0.9316,
                0.9287,
                0.9287,
                0.9287,
                0.4939,
                0.9287,
                0.9316,
                2.2695,
                0.1075,
                0.9287,
                0.9287,
                0.9287,
                0.9287,
                0.9321,
                0.9287,
            ]
        )
        torch.testing.assert_close(out.logits.float().cpu().mean(-1), expected_mean, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            out.logits[0, -1, :20].float().cpu(),
            expected_last_logits,
            rtol=1e-3,
            atol=1e-3,
        )

    @slow
    def test_generate_produces_audio(self):
        """Test that generate produces deterministic semantic tokens and audio from converted weights."""
        model = self._get_model()

        input_ids = torch.tensor([[1, 306, 4658, 278, 6593, 310]], device=torch_device)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        output = model.generate(input_ids=input_ids, max_new_tokens=5, top_k=1)

        self.assertIsNotNone(output.audio)
        self.assertEqual(len(output.audio), 1)
        self.assertGreater(output.audio[0].numel(), 0)
        self.assertEqual(output.semantic_tokens.shape, (1, 5))
        self.assertEqual(output.acoustic_values.shape, (1, 5, 36))
        self.assertTrue(torch.equal(output.semantic_tokens.cpu(), torch.tensor([[5980, 5980, 5980, 5980, 5980]])))

        expected_acoustic_tail = torch.tensor(
            [-1.4766, 0.3071, -1.1709, -0.7568, 1.4160, -0.8506, 0.0724, 0.5459, -1.0312, 1.9434]
        )
        expected_audio = torch.tensor(
            [
                -5.4264e-04,
                -1.4496e-03,
                -2.5272e-03,
                -3.3607e-03,
                -3.4122e-03,
                -2.7237e-03,
                -1.7328e-03,
                -8.9598e-04,
                -2.3019e-04,
                -5.0724e-05,
                -5.0163e-04,
                -1.3714e-03,
                -2.2926e-03,
                -2.4929e-03,
                -2.2831e-03,
                -2.1725e-03,
                -2.4471e-03,
                -3.1719e-03,
                -3.6354e-03,
                -3.8662e-03,
            ]
        )

        torch.testing.assert_close(
            output.acoustic_values[0, -1, :10].float().cpu(),
            expected_acoustic_tail,
            rtol=1e-3,
            atol=1e-3,
        )
        torch.testing.assert_close(output.audio[0][:20].float().cpu(), expected_audio, rtol=1e-3, atol=1e-4)
