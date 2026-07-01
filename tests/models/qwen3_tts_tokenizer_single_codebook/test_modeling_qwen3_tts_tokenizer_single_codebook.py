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
import tempfile
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin


if is_torch_available():
    import numpy as np
    import torch

    from transformers.models.qwen3_tts_tokenizer_single_codebook import (
        Qwen3TTSTokenizerSingleCodebookConfig,
        Qwen3TTSTokenizerSingleCodebookFeatureExtractor,
        Qwen3TTSTokenizerSingleCodebookModel,
    )


@require_torch
class Qwen3TTSTokenizerSingleCodebookModelTester:
    def __init__(self, parent, batch_size=2, feature_length=16, raw_audio_length=10):
        self.parent = parent
        self.batch_size = batch_size
        self.feature_length = feature_length
        self.raw_audio_length = raw_audio_length

        self.encoder_config = {
            "n_mels": 128,
            "n_ctx": 32,
            "n_state": 16,
            "n_head": 2,
            "n_layer": 1,
            "n_window": 8,
            "output_dim": 16,
            "audio_vq_codebook_size": 8,
            "audio_vq_codebook_dim": 8,
            "audio_vq_ds_rate": 2,
        }
        self.decoder_config = {
            "dit_config": {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "ff_mult": 1,
                "emb_dim": 16,
                "head_dim": 8,
                "repeats": 1,
                "num_embeds": 16,
                "mel_dim": 8,
                "block_size": 4,
                "look_ahead_layers": [],
                "look_backward_layers": [],
                "enc_emb_dim": 8,
                "enc_dim": 8,
                "enc_channels": [8],
                "enc_kernel_sizes": [1],
                "enc_dilations": [1],
                "enc_attention_channels": 4,
                "enc_se_channels": 4,
            },
            "bigvgan_config": {
                "mel_dim": 8,
                "upsample_initial_channel": 8,
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_rates": [2],
                "upsample_kernel_sizes": [4],
                "resblock_causal_types": ["1"],
            },
        }

    def get_config(self):
        return Qwen3TTSTokenizerSingleCodebookConfig(
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_features = torch.zeros(
            self.batch_size,
            config.encoder_config.n_mels,
            self.feature_length,
            device=torch_device,
        )
        feature_attention_mask = torch.ones(
            self.batch_size, self.feature_length, dtype=torch.long, device=torch_device
        )
        input_values = torch.zeros(self.batch_size, self.raw_audio_length, device=torch_device)
        padding_mask = torch.ones(self.batch_size, self.raw_audio_length, dtype=torch.long, device=torch_device)
        inputs_dict = {
            "input_values": input_values,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "padding_mask": padding_mask,
        }
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


@require_torch
class Qwen3TTSTokenizerSingleCodebookModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3TTSTokenizerSingleCodebookModel,) if is_torch_available() else ()
    _is_composite = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Qwen3TTSTokenizerSingleCodebookModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Qwen3TTSTokenizerSingleCodebookConfig, has_text_modality=False
        )
        _no_forward_tests = (
            "test_eager_matches_sdpa_inference",
            "test_attention_outputs",
            "test_hidden_states_output",
            "test_retain_grad_hidden_states_attentions",
            "test_model_forward_default_config_values",
            "test_feed_forward_chunking",
            "test_inputs_embeds",
        )
        if any(name in self._testMethodName for name in _no_forward_tests):
            self.skipTest("Qwen3TTSTokenizerSingleCodebookModel uses custom encode/decode methods")

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_instantiation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Qwen3TTSTokenizerSingleCodebookModel(config)
        self.assertIsNotNone(model)

    def test_save_load(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Qwen3TTSTokenizerSingleCodebookModel(config).eval().to(torch_device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            loaded = Qwen3TTSTokenizerSingleCodebookModel.from_pretrained(tmpdirname).eval().to(torch_device)
        for key in model.state_dict():
            self.assertTrue(
                torch.allclose(model.state_dict()[key], loaded.state_dict()[key]),
                f"Mismatch in key: {key}",
            )

    def test_feature_extractor_outputs_model_inputs(self):
        feature_extractor = Qwen3TTSTokenizerSingleCodebookFeatureExtractor(audio_vq_ds_rate=2)
        raw_audio = [
            np.zeros(321, dtype=np.float32),
            np.zeros(640, dtype=np.float32),
        ]

        inputs = feature_extractor(raw_audio, sampling_rate=16000, return_tensors="pt")

        self.assertEqual(
            set(inputs.keys()), {"input_features", "feature_attention_mask", "input_values", "padding_mask"}
        )
        self.assertEqual(inputs["input_features"].shape[0], 2)
        self.assertEqual(inputs["input_features"].shape[1], feature_extractor.feature_size)
        self.assertEqual(inputs["feature_attention_mask"].shape[0], 2)
        self.assertEqual(inputs["input_values"].shape, (2, 640))
        self.assertEqual(inputs["padding_mask"].sum(dim=-1).tolist(), [321, 640])

        # Waveforms are padded to hop_length * 2 * audio_vq_ds_rate before Whisper feature extraction.
        self.assertTrue((inputs["feature_attention_mask"].sum(dim=-1) > 0).all())

    def test_encode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerSingleCodebookModel(config).eval().to(torch_device)

        class DummyXVectorExtractor:
            def extract_code(self, wav):
                return np.ones(192, dtype=np.float32) * len(wav), np.ones((3, 80), dtype=np.float32)

        model.encoder_xvector_extractor = DummyXVectorExtractor()

        inputs_dict["padding_mask"] = torch.tensor(
            [
                [1] * 7 + [0] * 3,
                [1] * 10,
            ],
            dtype=torch.long,
            device=torch_device,
        )

        with torch.no_grad():
            outputs = model.encode(**inputs_dict)

        self.assertEqual(len(outputs.audio_codes), 2)
        self.assertEqual(len(outputs.xvectors), 2)
        self.assertEqual(len(outputs.ref_mels), 2)
        self.assertTrue(torch.all(outputs.xvectors[0] == 7))
        self.assertTrue(torch.all(outputs.xvectors[1] == 10))

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel has no standard forward")
    def test_training(self):
        pass

    @unittest.skip(reason="Batching equivalence is not applicable for variable-length codec outputs")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="No standard model outputs equivalence for codec models")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="No standard get_input_embeddings for codec model")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="No standard generate() for codec model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Composite model - base model prefix test not applicable")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="Determinism is not guaranteed across runs for codec models")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Compile not yet supported")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Flash attention right-padding equivalence is not applicable")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="No standard forward - all_tensors test not applicable")
    def test_all_tensors_are_parameter_or_buffer(self):
        pass

    @unittest.skip(reason="main_input_name requires converter re-run to propagate to modeling file")
    def test_model_main_input_name(self):
        pass

    @unittest.skip(reason="Composite config attn implementation is not propagated uniformly across sub-configs")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Codec model has no tied weights")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Codebook buffers are not reinitializable on meta device")
    def test_can_init_all_missing_weights(self):
        pass

    @unittest.skip(reason="Codebook buffers are not reinitializable on meta device")
    def test_init_weights_can_init_buffers(self):
        pass

    @unittest.skip(reason="No standard forward - left padding test not applicable")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="No standard forward - torch fx not applicable")
    def test_torch_fx(self):
        pass

    @unittest.skip(reason="No standard forward - torch fx not applicable")
    def test_torch_fx_output_loss(self):
        pass

    @unittest.skip(reason="No standard forward - sdpa dispatch not applicable")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Composite attention dispatch test assumes a VLM-style composite model")
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(reason="Torch export is not supported for this codec model")
    def test_torch_export(self):
        pass

    @unittest.skip(reason="Gradient checkpointing training coverage is not applicable for this codec model")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Gradient checkpointing training coverage is not applicable for this codec model")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Gradient checkpointing training coverage is not applicable for this codec model")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass
