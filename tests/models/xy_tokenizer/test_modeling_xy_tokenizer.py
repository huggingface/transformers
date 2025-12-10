# coding=utf-8
# Copyright 2025 OpenMOSS and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch XY-Tokenizer model."""

import inspect
import os
import tempfile
import unittest

import numpy as np

from transformers import XYTokenizerConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.testing_utils import (
    is_torch_available,
    require_torch,
    require_torchaudio,
    torch_device,
)
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor


if is_torch_available():
    import torch

    from transformers import XYTokenizer
    from transformers.models.xy_tokenizer.modeling_xy_tokenizer import (
        XYTokenizerDecoderOutput,
        XYTokenizerEncoderOutput,
    )


@require_torch
class XYTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        num_channels=1,
        sample_rate=16000,
        codebook_size=1024,
        num_quantizers=8,
        num_samples=16000,
        is_training=False,
        semantic_encoder_d_model=128,
        acoustic_encoder_d_model=128,
        code_dim=128,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.num_samples = num_samples
        self.is_training = is_training
        self.semantic_encoder_d_model = semantic_encoder_d_model
        self.acoustic_encoder_d_model = acoustic_encoder_d_model
        self.code_dim = code_dim

    def prepare_config_and_inputs(self):
        # Create audio input as mel-spectrogram features
        # XY-Tokenizer expects mel features, not raw audio
        num_mel_bins = 128  # Must match semantic_encoder_kwargs num_mel_bins
        time_steps = 6  # Small time steps for testing
        input_values = floats_tensor([self.batch_size, num_mel_bins, time_steps])

        config = self.get_config()
        return config, input_values

    def get_config(self):
        # Create config with nested kwargs structure that XY-Tokenizer expects
        params = {
            "semantic_encoder_kwargs": {
                "num_mel_bins": 128,
                "sampling_rate": self.sample_rate,
                "hop_length": 160,
                "stride_size": 2,
                "kernel_size": 3,
                "d_model": self.semantic_encoder_d_model,
                "encoder_layers": 2,  # Small for testing
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
            },
            "semantic_encoder_adapter_kwargs": {
                "input_dim": self.semantic_encoder_d_model,  # Input dimension from semantic encoder
                "d_model": self.semantic_encoder_d_model,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
                "output_dim": self.semantic_encoder_d_model,
            },
            "acoustic_encoder_kwargs": {
                "num_mel_bins": 128,
                "sampling_rate": self.sample_rate,
                "hop_length": 160,
                "stride_size": 2,
                "kernel_size": 3,
                "d_model": self.acoustic_encoder_d_model,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
            },
            "pre_rvq_adapter_kwargs": {
                "input_dim": self.semantic_encoder_d_model + self.acoustic_encoder_d_model,  # Concatenated dimension
                "d_model": self.code_dim,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
                "output_dim": self.code_dim,
            },
            "downsample_kwargs": {
                "d_model": self.code_dim,  # Should match pre_rvq_adapter output
                "avg_pooler": 2,
            },
            "quantizer_kwargs": {
                "num_quantizers": self.num_quantizers,
                "input_dim": self.code_dim * 2,  # downsample outputs intermediate_dim = d_model * avg_pooler
                "rvq_dim": self.code_dim,
                "output_dim": self.code_dim,  # Output back to code_dim for post_rvq_adapter
                "codebook_size": self.codebook_size,
                "codebook_dim": self.code_dim,
            },
            "post_rvq_adapter_kwargs": {
                "input_dim": self.code_dim,  # Input from quantizer
                "d_model": self.code_dim,
                "encoder_layers": 2,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 256,
                "output_dim": self.code_dim * 4,  # Output stride * d_model for upsample
            },
            "upsample_kwargs": {
                "d_model": self.code_dim,
                "stride": 4,
            },
            "acoustic_decoder_kwargs": {
                "num_mel_bins": 128,
                "sampling_rate": self.sample_rate,
                "hop_length": 160,
                "stride_size": 2,
                "kernel_size": 3,
                "d_model": self.code_dim,
                "decoder_layers": 2,
                "decoder_attention_heads": 4,
                "decoder_ffn_dim": 256,
            },
            "vocos_kwargs": {
                "input_channels": self.code_dim,
                "dim": 512,
                "intermediate_dim": 2048,
                "num_layers": 4,
                "n_fft": 640,
                "hop_size": 160,
            },
            "feature_extractor_kwargs": {
                "feature_size": 80,
                "sampling_rate": self.sample_rate,
                "hop_length": 160,
                "chunk_length": 30,
                "n_fft": 400,
            },
        }

        return XYTokenizerConfig(
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            code_dim=self.code_dim,
            sampling_rate=self.sample_rate,
            semantic_encoder_d_model=self.semantic_encoder_d_model,
            acoustic_encoder_d_model=self.acoustic_encoder_d_model,
            params=params,
        )

    def create_and_check_model_forward(self, config, input_values):
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()

        # XY tokenizer expects input_values directly in forward pass
        input_values = input_values.to(torch_device)
        attention_mask = torch.ones(self.batch_size, input_values.shape[-1]).to(torch_device)

        with torch.no_grad():
            result = model(input_values=input_values, attention_mask=attention_mask)

        # Check output structure - XY tokenizer can return various output types
        if isinstance(result, list):
            # XY tokenizer returns a list of tensors
            self.parent.assertIsInstance(result, list)
            self.parent.assertEqual(len(result), self.batch_size)
            for item in result:
                self.parent.assertIsInstance(item, torch.Tensor)
        elif hasattr(result, "audio_values"):
            # XYTokenizerOutput or XYTokenizerDecoderOutput
            self.parent.assertIsNotNone(result.audio_values)
            if hasattr(result, "audio_codes"):
                self.parent.assertIsNotNone(result.audio_codes)

    def create_and_check_encode(self, config, input_values):
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()

        # Create BatchFeature with attention mask
        attention_mask = torch.ones(self.batch_size, input_values.shape[-1]).to(torch_device)
        batch_feature = BatchFeature(
            {"input_features": input_values.to(torch_device), "attention_mask": attention_mask}
        )

        with torch.no_grad():
            result = model.encode(batch_feature)

        # Check encode output
        self.parent.assertIsInstance(result, XYTokenizerEncoderOutput)
        self.parent.assertIsNotNone(result.audio_codes)
        self.parent.assertIsNotNone(result.quantized_representation)
        self.parent.assertEqual(result.audio_codes.shape[1], self.batch_size)

    def prepare_config_and_inputs_for_common(self):
        config, input_values = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config, input_values = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict


@require_torch
@require_torchaudio
class XYTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    """Test suite for XY-Tokenizer model."""

    all_model_classes = (XYTokenizer,) if is_torch_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False
    test_can_init_all_missing_weights = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        """Override to remove attention/hidden states outputs."""
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = XYTokenizerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=XYTokenizerConfig, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            # XY_Tokenizer has different forward signature than expected
            expected_arg_names = ["input_values", "attention_mask", "n_quantizers"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(config)
            self.assertTrue(model.is_gradient_checkpointing)

    @unittest.skip("XYTokenizer cannot be tested with meta device")
    def test_can_load_with_meta_device_context_manager(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The XYTokenizer does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # Handle both tensor and list outputs
            if isinstance(first, list) and isinstance(second, list):
                # XY_Tokenizer outputs lists of tensors
                for f, s in zip(first, second):
                    out_1 = f.cpu().numpy()
                    out_2 = s.cpu().numpy()
                    out_1 = out_1[~np.isnan(out_1)]
                    out_2 = out_2[~np.isnan(out_2)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
            else:
                # Regular tensor outputs
                out_1 = first.cpu().numpy()
                out_2 = second.cpu().numpy()
                out_1 = out_1[~np.isnan(out_1)]
                out_2 = out_2[~np.isnan(out_2)]
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    @unittest.skip(reason="XY Tokenizer has special output format - returns list instead of dict/tuple")
    def test_model_outputs_equivalence(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv"]

                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of {model_class.__name__} seems not properly initialized",
                        )

    def test_batching_equivalence(self):
        """Test that batched and single inputs produce equivalent results."""
        config, input_values = self.model_tester.prepare_config_and_inputs()
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()

        # Test with single input
        single_input = input_values[:1]
        single_mask = torch.ones(1, single_input.shape[-1]).to(torch_device)
        single_batch = BatchFeature({"input_features": single_input.to(torch_device), "attention_mask": single_mask})

        with torch.no_grad():
            single_output = model.encode(single_batch)

        # Test with batched input (repeat the same input)
        batch_input = input_values
        batch_mask = torch.ones(self.model_tester.batch_size, batch_input.shape[-1]).to(torch_device)
        batch_feature = BatchFeature({"input_features": batch_input.to(torch_device), "attention_mask": batch_mask})

        with torch.no_grad():
            batch_output = model.encode(batch_feature)

        # Check first batch element matches single output
        self.assertTrue(
            torch.allclose(
                single_output.quantized_representation, batch_output.quantized_representation[:1], atol=1e-4
            )
        )

    def test_save_load(self):
        """Override to handle XY Tokenizer's list output format."""

        def check_save_load(out1, out2):
            # Handle list outputs from XY Tokenizer
            if isinstance(out1, list) and isinstance(out2, list):
                for o1, o2 in zip(out1, out2):
                    # make sure we don't have nans
                    out_2 = o2.cpu().numpy()
                    out_2[np.isnan(out_2)] = 0
                    out_2 = out_2[~np.isneginf(out_2)]

                    out_1 = o1.cpu().numpy()
                    out_1[np.isnan(out_1)] = 0
                    out_1 = out_1[~np.isneginf(out_1)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
            else:
                # Regular tensor outputs
                out_2 = out2.cpu().numpy()
                out_2[np.isnan(out_2)] = 0
                out_2 = out_2[~np.isneginf(out_2)]

                out_1 = out1.cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                out_1 = out_1[~np.isneginf(out_1)]
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

                # Save and load second time because `from_pretrained` adds a bunch of new config fields
                # so we need to make sure those fields can be loaded back after saving
                # Simply init as `model(config)` doesn't add those fields
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)

            check_save_load(first, second)

    def test_encode(self):
        """Test XY-Tokenizer encoding."""
        config, input_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encode(config, input_values)

    def test_encode_decode_consistency(self):
        """Test that encode->decode produces reasonable output."""
        config, input_values = self.model_tester.prepare_config_and_inputs()

        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()

        # Create BatchFeature
        attention_mask = torch.ones(self.model_tester.batch_size, input_values.shape[-1]).to(torch_device)
        batch_feature = BatchFeature(
            {"input_features": input_values.to(torch_device), "attention_mask": attention_mask}
        )

        with torch.no_grad():
            # Encode
            encode_result = model.encode(batch_feature)
            # Decode
            decode_result = model.decode(encode_result.audio_codes)

        # Check consistency - decode_result.audio_values is a list
        if isinstance(decode_result.audio_values, list):
            self.assertEqual(len(decode_result.audio_values), self.model_tester.batch_size)
            for audio in decode_result.audio_values:
                self.assertIsInstance(audio, torch.Tensor)
        else:
            self.assertEqual(decode_result.audio_values.shape[0], self.model_tester.batch_size)
            self.assertIsInstance(decode_result.audio_values, torch.Tensor)

    def test_audio_codes_properties(self):
        """Test properties of generated audio codes."""
        config, input_values = self.model_tester.prepare_config_and_inputs()

        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()

        # Create BatchFeature
        attention_mask = torch.ones(self.model_tester.batch_size, input_values.shape[-1]).to(torch_device)
        batch_feature = BatchFeature(
            {"input_features": input_values.to(torch_device), "attention_mask": attention_mask}
        )

        with torch.no_grad():
            result = model.encode(batch_feature)

        # Check audio codes are discrete
        self.assertEqual(result.audio_codes.dtype, torch.long)

        # Check codes are within valid range
        self.assertTrue(torch.all(result.audio_codes >= 0))
        self.assertTrue(torch.all(result.audio_codes < config.codebook_size))

        # Check number of quantizers
        self.assertEqual(result.audio_codes.shape[0], config.num_quantizers)

    def test_flash_attn_2_inference_equivalence_right_padding(self):
        """Test Flash Attention 2 with right padding for XY tokenizer."""
        # XY tokenizer supports right padding with Flash Attention 2
        # Use the base class implementation
        super().test_flash_attn_2_inference_equivalence_right_padding()

    @unittest.skip(reason="The XYTokenizer does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass


# Dataclass tests
@require_torch
class XYTokenizerDataClassesTest(unittest.TestCase):
    """Test XY-Tokenizer output dataclasses."""

    def test_encode_output(self):
        """Test XYTokenizerEncoderOutput dataclass."""
        batch_size, seq_len, num_quantizers = 2, 100, 8
        codebook_size = 256

        output = XYTokenizerEncoderOutput(
            quantized_representation=torch.randn(batch_size, 128, seq_len),
            audio_codes=torch.randint(0, codebook_size, (num_quantizers, batch_size, seq_len)),
            codes_lengths=torch.tensor([seq_len, seq_len]),
            commit_loss=torch.tensor(0.5),
            overlap_seconds=1,
        )

        self.assertEqual(output.quantized_representation.shape, (batch_size, 128, seq_len))
        self.assertEqual(output.audio_codes.shape, (num_quantizers, batch_size, seq_len))

    def test_decode_output(self):
        """Test XYTokenizerDecoderOutput dataclass."""
        batch_size, audio_len = 2, 16000

        output = XYTokenizerDecoderOutput(
            audio_values=torch.randn(batch_size, 1, audio_len),
            output_length=torch.tensor([audio_len, audio_len]),
        )

        self.assertEqual(output.audio_values.shape, (batch_size, 1, audio_len))
        self.assertEqual(output.output_length.shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
