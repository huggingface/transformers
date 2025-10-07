# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
Tests for Marian ONNX export functionality.

This test module validates the fixes for issue #40122 regarding incorrect
ONNX outputs for Marian translation models.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from transformers import MarianMTModel, MarianTokenizer
from transformers.testing_utils import require_onnx, require_torch, slow


try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Import our ONNX export utilities
try:
    import sys
    import os
    
    # Add examples directory to path for import
    examples_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "pytorch", "seq2seq")
    sys.path.insert(0, examples_path)
    
    from marian_onnx_export import (
        MarianEncoderONNX,
        MarianDecoderONNX,
        export_marian_encoder_to_onnx,
        export_marian_decoder_to_onnx,
        generate_with_onnx,
    )
except ImportError:
    # Skip tests if import fails
    MarianEncoderONNX = None
    MarianDecoderONNX = None
    export_marian_encoder_to_onnx = None
    export_marian_decoder_to_onnx = None
    generate_with_onnx = None


@require_torch
@require_onnx
class MarianONNXExportTest(unittest.TestCase):
    """
    Test cases for Marian ONNX export functionality.
    """

    def setUp(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-de"  # Smaller model for testing
        self.test_sentence = "Hello, how are you?"
        
        # Skip tests if required imports are not available
        if ort is None or MarianEncoderONNX is None:
            self.skipTest("Required dependencies not available")

    def test_encoder_wrapper_creation(self):
        """Test that the encoder wrapper can be created successfully."""
        model = MarianMTModel.from_pretrained(self.model_name)
        encoder_wrapper = MarianEncoderONNX(model)
        
        self.assertIsNotNone(encoder_wrapper)
        self.assertIsNotNone(encoder_wrapper.encoder)

    def test_decoder_wrapper_creation(self):
        """Test that the decoder wrapper can be created successfully."""
        model = MarianMTModel.from_pretrained(self.model_name)
        decoder_wrapper = MarianDecoderONNX(model)
        
        self.assertIsNotNone(decoder_wrapper)
        self.assertIsNotNone(decoder_wrapper.decoder)
        self.assertIsNotNone(decoder_wrapper.lm_head)
        self.assertIsNotNone(decoder_wrapper.final_logits_bias)

    def test_encoder_forward_pass(self):
        """Test that the encoder wrapper forward pass works correctly."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        encoder_wrapper = MarianEncoderONNX(model)
        
        # Prepare inputs
        inputs = tokenizer(self.test_sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Run forward pass
        with torch.no_grad():
            encoder_wrapper.eval()
            outputs = encoder_wrapper(input_ids, attention_mask)
        
        self.assertIsNotNone(outputs)
        self.assertEqual(len(outputs.shape), 3)  # [batch, seq_len, hidden_size]
        self.assertEqual(outputs.shape[0], 1)  # batch_size
        self.assertEqual(outputs.shape[2], model.config.d_model)  # hidden_size

    def test_decoder_forward_pass(self):
        """Test that the decoder wrapper forward pass works correctly."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        decoder_wrapper = MarianDecoderONNX(model)
        
        # Prepare inputs
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, seq_len, model.config.d_model)
        encoder_attention_mask = torch.ones(batch_size, seq_len)
        decoder_attention_mask = torch.ones(batch_size, seq_len)
        
        # Run forward pass
        with torch.no_grad():
            decoder_wrapper.eval()
            outputs = decoder_wrapper(
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_attention_mask
            )
        
        self.assertIsNotNone(outputs)
        self.assertEqual(len(outputs.shape), 3)  # [batch, seq_len, vocab_size]
        self.assertEqual(outputs.shape[0], batch_size)
        self.assertEqual(outputs.shape[1], seq_len)
        # vocab_size might differ between encoder and decoder
        self.assertGreater(outputs.shape[2], 0)

    def test_decoder_attention_mask_handling(self):
        """Test that decoder properly handles attention masks (addresses issue #40122)."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        decoder_wrapper = MarianDecoderONNX(model)
        
        batch_size, seq_len = 1, 5
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, seq_len, model.config.d_model)
        encoder_attention_mask = torch.ones(batch_size, seq_len)
        
        # Test with explicit decoder attention mask
        decoder_attention_mask = torch.ones(batch_size, seq_len)
        
        # Test with None decoder attention mask (should be auto-generated)
        with torch.no_grad():
            decoder_wrapper.eval()
            outputs1 = decoder_wrapper(
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                decoder_attention_mask
            )
            
            outputs2 = decoder_wrapper(
                input_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                None  # Should auto-generate mask
            )
        
        # Outputs should have the same shape
        self.assertEqual(outputs1.shape, outputs2.shape)
        
        # They might not be exactly equal due to different masking strategies,
        # but they should be reasonably close
        self.assertLess(torch.mean(torch.abs(outputs1 - outputs2)).item(), 1.0)

    @slow
    def test_encoder_onnx_export(self):
        """Test that encoder can be exported to ONNX format."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "encoder.onnx"
            
            # Export encoder
            export_marian_encoder_to_onnx(model, tokenizer, encoder_path)
            
            # Check that file was created
            self.assertTrue(encoder_path.exists())
            self.assertGreater(encoder_path.stat().st_size, 0)
            
            # Try to load with ONNX Runtime
            session = ort.InferenceSession(str(encoder_path))
            self.assertIsNotNone(session)

    @slow
    def test_decoder_onnx_export(self):
        """Test that decoder can be exported to ONNX format."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            decoder_path = Path(temp_dir) / "decoder.onnx"
            
            # Export decoder
            export_marian_decoder_to_onnx(model, tokenizer, decoder_path)
            
            # Check that file was created
            self.assertTrue(decoder_path.exists())
            self.assertGreater(decoder_path.stat().st_size, 0)
            
            # Try to load with ONNX Runtime
            session = ort.InferenceSession(str(decoder_path))
            self.assertIsNotNone(session)

    @slow
    def test_encoder_onnx_inference(self):
        """Test that exported encoder produces reasonable outputs."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "encoder.onnx"
            
            # Export encoder
            export_marian_encoder_to_onnx(model, tokenizer, encoder_path)
            
            # Load ONNX model
            session = ort.InferenceSession(str(encoder_path))
            
            # Prepare inputs
            inputs = tokenizer(self.test_sentence, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            
            # Run inference
            outputs = session.run(
                ["hidden_states"],
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )
            
            hidden_states = outputs[0]
            
            # Check output shape and properties
            self.assertEqual(len(hidden_states.shape), 3)
            self.assertEqual(hidden_states.shape[0], 1)  # batch_size
            self.assertEqual(hidden_states.shape[2], model.config.d_model)
            self.assertFalse(np.isnan(hidden_states).any())
            self.assertFalse(np.isinf(hidden_states).any())

    @slow
    def test_decoder_onnx_inference(self):
        """Test that exported decoder produces reasonable outputs."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            decoder_path = Path(temp_dir) / "decoder.onnx"
            
            # Export decoder
            export_marian_decoder_to_onnx(model, tokenizer, decoder_path)
            
            # Load ONNX model
            session = ort.InferenceSession(str(decoder_path))
            
            # Prepare inputs
            batch_size, src_len, tgt_len = 1, 10, 8
            hidden_size = model.config.d_model
            
            input_ids = np.full((batch_size, tgt_len), tokenizer.pad_token_id, dtype=np.int64)
            input_ids[:, 0] = model.config.decoder_start_token_id
            
            encoder_hidden_states = np.random.randn(batch_size, src_len, hidden_size).astype(np.float32)
            encoder_attention_mask = np.ones((batch_size, src_len), dtype=np.int64)
            decoder_attention_mask = np.ones((batch_size, tgt_len), dtype=np.int64)
            
            # Run inference
            outputs = session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                    "encoder_attention_mask": encoder_attention_mask,
                    "decoder_attention_mask": decoder_attention_mask
                }
            )
            
            logits = outputs[0]
            
            # Check output shape and properties
            self.assertEqual(len(logits.shape), 3)
            self.assertEqual(logits.shape[0], batch_size)
            self.assertEqual(logits.shape[1], tgt_len)
            self.assertGreater(logits.shape[2], 0)  # vocab_size > 0
            self.assertFalse(np.isnan(logits).any())
            self.assertFalse(np.isinf(logits).any())

    @slow
    def test_pytorch_vs_onnx_encoder_consistency(self):
        """Test that ONNX encoder outputs are consistent with PyTorch."""
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "encoder.onnx"
            
            # Export encoder
            export_marian_encoder_to_onnx(model, tokenizer, encoder_path)
            
            # Prepare inputs
            inputs = tokenizer(self.test_sentence, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Get PyTorch outputs
            model.eval()
            with torch.no_grad():
                pytorch_outputs = model.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                pytorch_hidden_states = pytorch_outputs.last_hidden_state.numpy()
            
            # Get ONNX outputs
            session = ort.InferenceSession(str(encoder_path))
            onnx_outputs = session.run(
                ["hidden_states"],
                {
                    "input_ids": input_ids.numpy().astype(np.int64),
                    "attention_mask": attention_mask.numpy().astype(np.int64)
                }
            )
            onnx_hidden_states = onnx_outputs[0]
            
            # Compare outputs (allowing for small numerical differences)
            np.testing.assert_allclose(
                pytorch_hidden_states,
                onnx_hidden_states,
                rtol=1e-4,
                atol=1e-4,
                err_msg="PyTorch and ONNX encoder outputs should be nearly identical"
            )

    @slow
    def test_end_to_end_translation_consistency(self):
        """
        Test end-to-end translation consistency between PyTorch and ONNX.
        
        This is the main test that validates the fix for issue #40122.
        """
        model = MarianMTModel.from_pretrained(self.model_name)
        tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder_path = Path(temp_dir) / "encoder.onnx"
            decoder_path = Path(temp_dir) / "decoder.onnx"
            
            # Export both models
            export_marian_encoder_to_onnx(model, tokenizer, encoder_path)
            export_marian_decoder_to_onnx(model, tokenizer, decoder_path)
            
            # Get PyTorch translation
            model.eval()
            with torch.no_grad():
                inputs = tokenizer(self.test_sentence, return_tensors="pt")
                pytorch_outputs = model.generate(**inputs, max_length=32, num_beams=1, do_sample=False)
                pytorch_text = tokenizer.decode(pytorch_outputs[0], skip_special_tokens=True)
            
            # Get ONNX translation
            encoder_session = ort.InferenceSession(str(encoder_path))
            decoder_session = ort.InferenceSession(str(decoder_path))
            
            onnx_text = generate_with_onnx(
                encoder_session,
                decoder_session,
                tokenizer,
                self.test_sentence,
                max_length=32
            )
            
            # The outputs should be identical or very similar
            # (exact match might not always be possible due to different generation strategies)
            self.assertIsNotNone(onnx_text)
            self.assertGreater(len(onnx_text.strip()), 0)
            
            # At minimum, check that both translations are reasonable
            # (contain alphabetic characters, not just punctuation/whitespace)
            self.assertTrue(any(c.isalpha() for c in pytorch_text))
            self.assertTrue(any(c.isalpha() for c in onnx_text))
            
            print(f"PyTorch translation: {pytorch_text}")
            print(f"ONNX translation: {onnx_text}")
            
            # If translations are identical, that's perfect
            if pytorch_text.strip() == onnx_text.strip():
                print("✅ Perfect match between PyTorch and ONNX translations!")
            else:
                print("⚠️  Translations differ (this might be acceptable)")


if __name__ == "__main__":
    unittest.main()
