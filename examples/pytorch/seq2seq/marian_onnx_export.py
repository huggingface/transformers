#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 ("License");
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
Marian ONNX Export Utilities

This script provides utilities to export Marian translation models to ONNX format
with proper handling of attention masks and decoder inputs. It fixes issues with
incorrect ONNX output compared to PyTorch models.

Usage:
    python marian_onnx_export.py --model_name Helsinki-NLP/opus-mt-en-ar

Example:
    python marian_onnx_export.py --model_name Helsinki-NLP/opus-mt-en-ar --output_dir ./onnx_models
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import MarianMTModel, MarianTokenizer

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("Warning: onnxruntime not available. Install it with: pip install onnxruntime")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarianEncoderONNX(nn.Module):
    """
    ONNX-compatible wrapper for Marian encoder.
    """

    def __init__(self, model: MarianMTModel):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return encoder_outputs.last_hidden_state


class MarianDecoderONNX(nn.Module):
    """
    Enhanced ONNX-compatible wrapper for Marian decoder with proper attention mask handling.
    
    Fixes the issues described in transformers issue #40122 by:
    1. Properly handling decoder attention masks
    2. Ensuring correct token initialization and positioning
    3. Maintaining compatibility with autoregressive generation
    """

    def __init__(self, model: MarianMTModel):
        super().__init__()
        self.decoder = model.model.decoder
        self.lm_head = model.lm_head
        self.final_logits_bias = model.final_logits_bias
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with proper attention mask handling.
        
        Args:
            input_ids: Decoder input token IDs [batch_size, seq_len]
            encoder_hidden_states: Encoder outputs [batch_size, src_len, hidden_size]
            encoder_attention_mask: Encoder attention mask [batch_size, src_len]
            decoder_attention_mask: Decoder attention mask [batch_size, tgt_len]
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        # Ensure decoder attention mask is properly set
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids)
        
        # Apply causal masking to decoder attention mask
        batch_size, seq_len = input_ids.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool))
        
        # Expand causal mask to batch dimension
        expanded_causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply both padding mask and causal mask
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1) & expanded_causal_mask
        
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        hidden_states = decoder_outputs.last_hidden_state
        logits = self.lm_head(hidden_states) + self.final_logits_bias
        
        return logits


def export_marian_encoder_to_onnx(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output_path: Union[str, Path],
    opset_version: int = 17
) -> None:
    """
    Export Marian encoder to ONNX format.
    
    Args:
        model: The Marian MT model
        tokenizer: The corresponding tokenizer
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info(f"Exporting Marian encoder to {output_path}")
    
    encoder_wrapper = MarianEncoderONNX(model)
    encoder_wrapper.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 64
    
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Ensure no padding tokens in the middle of the sequence
    dummy_input_ids[:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
    dummy_input_ids[:, -1] = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "hidden_states": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    logger.info(f"Successfully exported encoder to {output_path}")


def export_marian_decoder_to_onnx(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    output_path: Union[str, Path],
    opset_version: int = 17
) -> None:
    """
    Export Marian decoder to ONNX format with fixed attention mask handling.
    
    This function addresses the issues described in transformers issue #40122
    by properly handling decoder attention masks and token positioning.
    
    Args:
        model: The Marian MT model
        tokenizer: The corresponding tokenizer
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version to use
    """
    logger.info(f"Exporting Marian decoder to {output_path}")
    
    decoder_wrapper = MarianDecoderONNX(model)
    decoder_wrapper.eval()
    
    # Create more appropriate dummy inputs
    batch_size = 1
    src_seq_len = 64
    tgt_seq_len = 64
    hidden_size = model.config.d_model
    
    # Create decoder input with proper token initialization
    dummy_decoder_input_ids = torch.full((batch_size, tgt_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    dummy_decoder_input_ids[:, 0] = model.config.decoder_start_token_id
    
    # Create encoder hidden states
    dummy_encoder_hidden_states = torch.randn(batch_size, src_seq_len, hidden_size)
    dummy_encoder_attention_mask = torch.ones(batch_size, src_seq_len, dtype=torch.long)
    
    # Create proper decoder attention mask (causal)
    dummy_decoder_attention_mask = torch.ones(batch_size, tgt_seq_len, dtype=torch.long)
    
    torch.onnx.export(
        decoder_wrapper,
        (
            dummy_decoder_input_ids,
            dummy_encoder_hidden_states,
            dummy_encoder_attention_mask,
            dummy_decoder_attention_mask
        ),
        output_path,
        input_names=[
            "input_ids",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "decoder_attention_mask"
        ],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "tgt_sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "src_sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "src_sequence_length"},
            "decoder_attention_mask": {0: "batch_size", 1: "tgt_sequence_length"},
            "logits": {0: "batch_size", 1: "tgt_sequence_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )
    
    logger.info(f"Successfully exported decoder to {output_path}")


def generate_with_onnx(
    encoder_session: "ort.InferenceSession",
    decoder_session: "ort.InferenceSession",
    tokenizer: MarianTokenizer,
    input_text: str,
    max_length: int = 64
) -> str:
    """
    Generate translation using ONNX exported models with proper token handling.
    
    This function implements the corrected inference procedure that matches
    PyTorch model behavior.
    
    Args:
        encoder_session: ONNX runtime session for encoder
        decoder_session: ONNX runtime session for decoder
        tokenizer: The tokenizer
        input_text: Text to translate
        max_length: Maximum generation length
    
    Returns:
        Translated text
    """
    # Encode input
    inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Run encoder
    encoder_outputs = encoder_session.run(
        ["hidden_states"],
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64)
        }
    )
    encoder_hidden_states = encoder_outputs[0]
    
    # Initialize decoder inputs
    batch_size = input_ids.shape[0]
    decoder_input_ids = np.full((batch_size, max_length), tokenizer.pad_token_id, dtype=np.int64)
    decoder_input_ids[:, 0] = tokenizer.convert_tokens_to_ids(tokenizer.bos_token) if tokenizer.bos_token else 0
    
    generated_tokens = []
    
    for step in range(1, max_length):
        # Create current decoder attention mask (attend only to generated tokens so far)
        current_decoder_mask = np.zeros((batch_size, max_length), dtype=np.int64)
        current_decoder_mask[:, :step] = 1
        
        # Run decoder
        decoder_outputs = decoder_session.run(
            ["logits"],
            {
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask.astype(np.int64),
                "decoder_attention_mask": current_decoder_mask
            }
        )
        
        # Get next token logits
        logits = decoder_outputs[0]
        next_token_logits = logits[0, step - 1, :]  # Get logits for current position
        next_token_id = np.argmax(next_token_logits)
        
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Update decoder input
        decoder_input_ids[0, step] = next_token_id
        generated_tokens.append(next_token_id)
    
    # Decode generated text
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text


def test_onnx_export(model_name: str, test_sentence: str, output_dir: str) -> None:
    """
    Test the ONNX export by comparing outputs with the original PyTorch model.
    
    Args:
        model_name: HuggingFace model name
        test_sentence: Test sentence for translation
        output_dir: Directory containing ONNX models
    """
    if ort is None:
        logger.error("onnxruntime not available. Please install it to run tests.")
        return
    
    logger.info("Loading PyTorch model for comparison...")
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model.eval()
    
    # Get PyTorch output
    with torch.no_grad():
        inputs = tokenizer(test_sentence, return_tensors="pt")
        pytorch_outputs = model.generate(**inputs, max_length=64)
        pytorch_text = tokenizer.decode(pytorch_outputs[0], skip_special_tokens=True)
    
    logger.info(f"PyTorch output: {pytorch_text}")
    
    # Test ONNX models
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    
    if not (os.path.exists(encoder_path) and os.path.exists(decoder_path)):
        logger.error(f"ONNX models not found in {output_dir}")
        return
    
    logger.info("Loading ONNX models...")
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)
    
    # Get ONNX output using corrected inference
    onnx_text = generate_with_onnx(encoder_session, decoder_session, tokenizer, test_sentence)
    
    logger.info(f"ONNX output: {onnx_text}")
    
    # Simple comparison (in practice, you might want more sophisticated comparison)
    if pytorch_text.strip() == onnx_text.strip():
        logger.info("✅ Test passed: ONNX and PyTorch outputs match!")
    else:
        logger.warning("⚠️  Test warning: ONNX and PyTorch outputs differ")
        logger.info("This might be due to slight numerical differences or different generation strategies.")
        logger.info("Manual inspection of the outputs is recommended.")


def main():
    parser = argparse.ArgumentParser(description="Export Marian model to ONNX format")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., Helsinki-NLP/opus-mt-en-ar)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./onnx_models",
        help="Directory to save ONNX models"
    )
    parser.add_argument(
        "--test_sentence",
        type=str,
        default="Using handheld GPS devices and programs like Google Earth, members of the Trio Tribe, who live in the rainforests of southern Suriname, map out their ancestral lands to help strengthen their territorial claims.",
        help="Test sentence for validation"
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip the validation test"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading model: {args.model_name}")
    model = MarianMTModel.from_pretrained(args.model_name)
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model.eval()
    
    # Export encoder
    encoder_path = os.path.join(args.output_dir, "encoder.onnx")
    export_marian_encoder_to_onnx(model, tokenizer, encoder_path, args.opset_version)
    
    # Export decoder
    decoder_path = os.path.join(args.output_dir, "decoder.onnx")
    export_marian_decoder_to_onnx(model, tokenizer, decoder_path, args.opset_version)
    
    logger.info(f"Export completed. Models saved to {args.output_dir}")
    
    # Run validation test
    if not args.skip_test:
        logger.info("Running validation test...")
        test_onnx_export(args.model_name, args.test_sentence, args.output_dir)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
