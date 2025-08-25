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

import tempfile
import unittest

import numpy as np

from transformers import XYTokenizerConfig, XYTokenizer, XYTokenizerFeatureExtractor
from transformers.testing_utils import require_torch, require_torchaudio, torch_device
from transformers.utils import is_torch_available
from transformers.models.xy_tokenizer.modeling_xy_tokenizer import (
    XYTokenizerEncodeOutput,
    XYTokenizerDecodeOutput, 
    XYTokenizerOutput,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch


class XYTokenizerModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_channels=1,
        sequence_length=1024,  # Audio sequence length
        input_sample_rate=16000,
        output_sample_rate=24000,
        encoder_downsample_rate=320,
        decoder_upsample_rate=320,
        code_dim=128,  # Smaller for testing
        semantic_encoder_d_model=128,
        acoustic_encoder_d_model=128,
        num_quantizers=8,  # Smaller for testing
        codebook_size=256,  # Smaller for testing
        semantic_codebook_size=256,
        commit_loss_lambda=0.25,
        bandwidth=6.0,
        sampling_rate=24000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.encoder_downsample_rate = encoder_downsample_rate
        self.decoder_upsample_rate = decoder_upsample_rate
        self.code_dim = code_dim
        self.semantic_encoder_d_model = semantic_encoder_d_model
        self.acoustic_encoder_d_model = acoustic_encoder_d_model
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.semantic_codebook_size = semantic_codebook_size
        self.commit_loss_lambda = commit_loss_lambda
        self.bandwidth = bandwidth
        self.sampling_rate = sampling_rate

    def prepare_config_and_inputs(self):
        # Create random audio input
        input_values = floats_tensor([self.batch_size, self.num_channels, self.sequence_length])
        
        config = self.get_config()
        return config, input_values

    def get_config(self):
        return XYTokenizerConfig(
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
            encoder_downsample_rate=self.encoder_downsample_rate,
            decoder_upsample_rate=self.decoder_upsample_rate,
            code_dim=self.code_dim,
            semantic_encoder_d_model=self.semantic_encoder_d_model,
            acoustic_encoder_d_model=self.acoustic_encoder_d_model,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            semantic_codebook_size=self.semantic_codebook_size,
            commit_loss_lambda=self.commit_loss_lambda,
            bandwidth=self.bandwidth,
            sampling_rate=self.sampling_rate,
        )

    def create_and_check_model(self, config, input_values):
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            result = model(input_values)
            
        # Check that we get XYTokenizerOutput
        self.parent.assertIsInstance(result, XYTokenizerOutput)
        self.parent.assertIsNotNone(result.audio_values)
        self.parent.assertIsNotNone(result.quantized_representation)
        self.parent.assertIsNotNone(result.audio_codes)
        
        # Check shapes
        self.parent.assertEqual(result.audio_values.shape[0], self.batch_size)
        self.parent.assertEqual(result.audio_codes.shape[1], self.batch_size)

    def create_and_check_encode(self, config, input_values):
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            result = model.encode(input_values)
            
        # Check that we get XYTokenizerEncodeOutput
        self.parent.assertIsInstance(result, XYTokenizerEncodeOutput)
        self.parent.assertIsNotNone(result.audio_codes)
        self.parent.assertIsNotNone(result.quantized_representation)
        
        # Check shapes
        self.parent.assertEqual(result.audio_codes.shape[1], self.batch_size)

    def create_and_check_decode(self, config, input_values):
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        # First encode to get codes
        with torch.no_grad():
            encode_result = model.encode(input_values)
            audio_codes = encode_result.audio_codes
            
            # Then decode
            decode_result = model.decode(audio_codes)
            
        # Check that we get XYTokenizerDecodeOutput
        self.parent.assertIsInstance(decode_result, XYTokenizerDecodeOutput)
        self.parent.assertIsNotNone(decode_result.audio_values)
        
        # Check shapes
        self.parent.assertEqual(decode_result.audio_values.shape[0], self.batch_size)

    def prepare_config_and_inputs_for_common(self):
        config, input_values = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict


@require_torch
@require_torchaudio
class XYTokenizerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (XYTokenizer,) if is_torch_available() else ()
    
    # Override some settings for audio model
    test_pruning = False
    test_missing_keys = False
    test_model_parallel = False
    test_head_masking = False
    test_resize_embeddings = False
    test_resize_tokens_embeddings = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = XYTokenizerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=XYTokenizerConfig, code_dim=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_values)

    def test_encode(self):
        config, input_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_encode(config, input_values)

    def test_decode(self):
        config, input_values = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_decode(config, input_values)

    def test_encode_decode_consistency(self):
        """Test that encode->decode produces reasonable output."""
        config, input_values = self.model_tester.prepare_config_and_inputs()
        
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            # Encode
            encode_result = model.encode(input_values)
            
            # Decode  
            decode_result = model.decode(encode_result.audio_codes)
            
            # Check that output has same batch size as input
            self.assertEqual(input_values.shape[0], decode_result.audio_values.shape[0])
            
            # Check that we have reasonable audio output
            self.assertIsInstance(decode_result.audio_values, torch.Tensor)
            self.assertEqual(decode_result.audio_values.ndim, 3)  # (batch, channels, time)

    def test_forward_pass_outputs(self):
        """Test that forward pass produces all expected outputs."""
        config, input_values = self.model_tester.prepare_config_and_inputs()
        
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(input_values)
            
        # Check all expected outputs are present
        self.assertIsNotNone(outputs.audio_values)
        self.assertIsNotNone(outputs.quantized_representation)
        self.assertIsNotNone(outputs.audio_codes)
        
        # Check output shapes are consistent
        batch_size = input_values.shape[0]
        self.assertEqual(outputs.audio_values.shape[0], batch_size)
        self.assertEqual(outputs.quantized_representation.shape[0], batch_size)
        self.assertEqual(outputs.audio_codes.shape[1], batch_size)

    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        config = self.model_tester.get_config()
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        # Test with different lengths
        for seq_len in [512, 1024, 2048]:
            input_values = torch.randn(1, 1, seq_len).to(torch_device)
            
            with torch.no_grad():
                outputs = model(input_values)
                
            self.assertEqual(outputs.audio_values.shape[0], 1)
            self.assertIsNotNone(outputs.audio_codes)

    def test_quantization_properties(self):
        """Test quantization-specific properties."""
        config, input_values = self.model_tester.prepare_config_and_inputs()
        
        model = XYTokenizer(config=config)
        model.to(torch_device)
        model.eval()
        
        with torch.no_grad():
            result = model.encode(input_values)
            
        # Audio codes should be discrete (integers)
        self.assertEqual(result.audio_codes.dtype, torch.long)
        
        # Check codes are within valid range
        self.assertTrue(torch.all(result.audio_codes >= 0))
        self.assertTrue(torch.all(result.audio_codes < config.codebook_size))
        
        # Check we have the expected number of codebooks
        self.assertEqual(result.audio_codes.shape[0], config.num_quantizers)


@require_torch
class XYTokenizerDataClassesTest(unittest.TestCase):
    """Test XY-Tokenizer output dataclasses."""

    def test_encode_output(self):
        """Test XYTokenizerEncodeOutput dataclass."""
        import torch
        
        batch_size, seq_len, num_quantizers = 2, 100, 8
        codebook_size = 256
        
        output = XYTokenizerEncodeOutput(
            quantized_representation=torch.randn(batch_size, 128, seq_len),
            audio_codes=torch.randint(0, codebook_size, (num_quantizers, batch_size, seq_len)),
            codes_lengths=torch.tensor([seq_len, seq_len]),
            commit_loss=torch.tensor(0.5),
            overlap_seconds=1
        )
        
        self.assertEqual(output.quantized_representation.shape, (batch_size, 128, seq_len))
        self.assertEqual(output.audio_codes.shape, (num_quantizers, batch_size, seq_len))
        self.assertEqual(output.codes_lengths.shape, (batch_size,))
        self.assertIsInstance(output.commit_loss, torch.Tensor)
        self.assertEqual(output.overlap_seconds, 1)

    def test_decode_output(self):
        """Test XYTokenizerDecodeOutput dataclass."""
        import torch
        
        batch_size, seq_len = 2, 16000
        
        output = XYTokenizerDecodeOutput(
            audio_values=torch.randn(batch_size, 1, seq_len),
            output_length=torch.tensor([seq_len, seq_len])
        )
        
        self.assertEqual(output.audio_values.shape, (batch_size, 1, seq_len))
        self.assertEqual(output.output_length.shape, (batch_size,))

    def test_full_output(self):
        """Test XYTokenizerOutput dataclass."""
        import torch
        
        batch_size, seq_len, num_quantizers = 2, 16000, 8
        code_seq_len = seq_len // 320  # Assuming downsample rate of 320
        
        output = XYTokenizerOutput(
            audio_values=torch.randn(batch_size, 1, seq_len),
            output_length=torch.tensor([seq_len, seq_len]),
            quantized_representation=torch.randn(batch_size, 128, code_seq_len),
            audio_codes=torch.randint(0, 256, (num_quantizers, batch_size, code_seq_len)),
            codes_lengths=torch.tensor([code_seq_len, code_seq_len])
        )
        
        self.assertEqual(output.audio_values.shape, (batch_size, 1, seq_len))
        self.assertEqual(output.quantized_representation.shape, (batch_size, 128, code_seq_len))
        self.assertEqual(output.audio_codes.shape, (num_quantizers, batch_size, code_seq_len))


@require_torch
@require_torchaudio
class XYTokenizerIntegrationTest(unittest.TestCase):
    """Integration tests for XY-Tokenizer."""

    def setUp(self):
        # Use small config for testing
        self.config = XYTokenizerConfig(
            input_sample_rate=16000,
            output_sample_rate=16000,  # Same for simplicity
            encoder_downsample_rate=320,
            decoder_upsample_rate=320,
            code_dim=64,
            num_quantizers=4,
            codebook_size=128,
            sampling_rate=16000,
        )

    def test_end_to_end_processing(self):
        """Test complete encode-decode cycle."""
        model = XYTokenizer(self.config)
        model.to(torch_device)
        model.eval()
        
        # Create test audio (1 second at 16kHz)
        input_audio = torch.randn(1, 1, 16000).to(torch_device)
        
        with torch.no_grad():
            # Test full forward pass
            outputs = model(input_audio)
            
            # Test encode only
            encode_outputs = model.encode(input_audio)
            
            # Test decode only
            decode_outputs = model.decode(encode_outputs.audio_codes)
            
        # Verify consistency between different calls
        self.assertTrue(torch.allclose(
            outputs.audio_codes, 
            encode_outputs.audio_codes, 
            atol=1e-6
        ))
        
        self.assertTrue(torch.allclose(
            outputs.audio_values, 
            decode_outputs.audio_values, 
            atol=1e-6
        ))

    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        model = XYTokenizer(self.config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model.save_pretrained(tmpdir)
            
            # Load model
            loaded_model = XYTokenizer.from_pretrained(tmpdir)
            
            # Test that loaded model works
            input_audio = torch.randn(1, 1, 8000)
            
            with torch.no_grad():
                original_output = model(input_audio)
                loaded_output = loaded_model(input_audio)
                
            # Outputs should be identical
            self.assertTrue(torch.allclose(
                original_output.audio_codes, 
                loaded_output.audio_codes,
                atol=1e-6
            ))


if __name__ == "__main__":
    unittest.main()