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
    import torch

    from transformers import Qwen3TTSTokenizerSingleCodebookConfig, Qwen3TTSTokenizerSingleCodebookModel


@require_torch
class Qwen3TTSTokenizerSingleCodebookModelTester:
    def __init__(self, parent, batch_size=2, feature_length=24, short_feature_length=16):
        self.parent = parent
        self.batch_size = batch_size
        self.feature_length = feature_length
        self.short_feature_length = short_feature_length

        self.encoder_config = {
            "num_mel_bins": 16,
            "hidden_size": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "max_source_positions": 64,
            "n_window": 4,
        }
        self.quantizer_config = {
            "hidden_size": 16,
            "codebook_size": 8,
            "codebook_dim": 8,
            "downsample_rate": 2,
        }
        self.decoder_config = {
            "dit_config": {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "ff_mult": 1,
                "emb_dim": 16,
                "head_dim": 8,
                "repeats": 2,
                "num_embeds": 16,
                "mel_dim": 8,
                "block_size": 4,
                "look_ahead_layers": [1],
                "look_backward_layers": [0],
                "enc_emb_dim": 8,
                "enc_dim": 8,
                "enc_channels": [8, 8, 8],
                "enc_kernel_sizes": [3, 3, 1],
                "enc_dilations": [1, 1, 1],
                "enc_attention_channels": 4,
                "enc_se_channels": 4,
            },
            "bigvgan_config": {
                "mel_dim": 8,
                "upsample_initial_channel": 16,
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3, 5]],
                "upsample_rates": [2, 2],
                "upsample_kernel_sizes": [4, 4],
                "resblock_causal_modes": ["full_causal", "hybrid"],
            },
        }
        # 2 mel frames per code (`repeats`) times the vocoder upsampling
        self.decode_upsample_rate = 2 * 2 * 2

    def get_config(self):
        return Qwen3TTSTokenizerSingleCodebookConfig(
            encoder_config=self.encoder_config,
            quantizer_config=self.quantizer_config,
            decoder_config=self.decoder_config,
            decode_upsample_rate=self.decode_upsample_rate,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_features = torch.randn(
            self.batch_size,
            config.encoder_config.num_mel_bins,
            self.feature_length,
            device=torch_device,
        )
        input_features_mask = torch.ones(self.batch_size, self.feature_length, dtype=torch.long, device=torch_device)
        input_features_mask[-1, self.short_feature_length :] = 0
        inputs_dict = {
            "input_features": input_features,
            "input_features_mask": input_features_mask,
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
            "test_can_capture_specific_layers_hidden_states",
            "test_retain_grad_hidden_states_attentions",
            "test_model_forward_default_config_values",
            "test_feed_forward_chunking",
            "test_inputs_embeds",
            "test_capture_outputs_decorator",
            "test_multi_gpu_data_parallel_forward",
        )
        if any(name in self._testMethodName for name in _no_forward_tests):
            self.skipTest("Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_instantiation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Qwen3TTSTokenizerSingleCodebookModel(config)
        self.assertTrue(hasattr(model, "encoder"))
        self.assertTrue(hasattr(model, "quantizer"))
        self.assertTrue(hasattr(model, "decoder"))

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

    def test_encode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerSingleCodebookModel(config).eval().to(torch_device)

        with torch.no_grad():
            outputs = model.encode(**inputs_dict)

        # one code per `downsample_rate` frames of the stride-2 convolution output
        expected_lengths = [
            ((length - 1) // 2 + 1) // config.quantizer_config.downsample_rate
            for length in inputs_dict["input_features_mask"].sum(-1).tolist()
        ]
        self.assertEqual(outputs.audio_codes.shape, (self.model_tester.batch_size, max(expected_lengths)))
        self.assertEqual(outputs.audio_codes_mask.sum(-1).tolist(), expected_lengths)
        self.assertTrue((outputs.audio_codes < config.quantizer_config.codebook_size).all())

    def test_encode_padded_item_matches_unpadded(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerSingleCodebookModel(config).eval().to(torch_device)
        # random codebook so that codes depend on the encoder output
        model.quantizer.vq.codebook.embed.normal_()

        short_length = self.model_tester.short_feature_length
        with torch.no_grad():
            batched = model.encode(**inputs_dict)
            single = model.encode(inputs_dict["input_features"][-1:, :, :short_length])

        num_codes = int(batched.audio_codes_mask[-1].sum())
        self.assertEqual(single.audio_codes.shape[1], num_codes)
        torch.testing.assert_close(batched.audio_codes[-1, :num_codes], single.audio_codes[0])

    def test_decode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        model = Qwen3TTSTokenizerSingleCodebookModel(config).eval().to(torch_device)
        dit_config = config.decoder_config.dit_config
        batch_size = self.model_tester.batch_size

        with torch.no_grad():
            encoded = model.encode(**inputs_dict)
            audio_codes = encoded.audio_codes.masked_fill(~encoded.audio_codes_mask, -1)
            xvectors = torch.randn(batch_size, dit_config.enc_emb_dim, device=torch_device)
            ref_mels = torch.randn(batch_size, 10, dit_config.mel_dim, device=torch_device)
            outputs = model.decode(audio_codes, xvectors=xvectors, ref_mels=ref_mels, num_steps=2)

        code_lengths = encoded.audio_codes_mask.sum(-1)
        expected_samples = int(code_lengths.max()) * config.decode_upsample_rate
        self.assertEqual(outputs.audio_values.shape, (batch_size, expected_samples))
        # the decoded waveform is zero past every item's own duration
        short_samples = int(code_lengths[-1]) * config.decode_upsample_rate
        self.assertLess(short_samples, expected_samples)
        self.assertTrue((outputs.audio_values[-1, short_samples:] == 0).all())
        self.assertTrue((outputs.audio_values[0] != 0).any())

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_capture_outputs_decorator(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_training(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Codec model has no input embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Codec model has no generate()")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Composite model with encoder, quantizer and decoder; there is no base model prefix")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_all_tensors_are_parameter_or_buffer(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_model_main_input_name(self):
        pass

    @unittest.skip(reason="The DiT and BigVGAN sub-models always run with sdpa; their block-causal mask is boolean")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Codec model has no tied weights")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_torch_fx(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_torch_fx_output_loss(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Composite attention dispatch test assumes a VLM-style composite model")
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_torch_export(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Qwen3TTSTokenizerSingleCodebookModel exposes encode/decode only, no forward")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@require_torch
class Qwen3TTSTokenizerSingleCodebookIntegrationTest(unittest.TestCase):
    @unittest.skip(reason="No public Qwen3-TTS-Tokenizer-25Hz checkpoint (QwenLM/Qwen3-TTS#34)")
    def test_parity_with_original(self):
        pass
