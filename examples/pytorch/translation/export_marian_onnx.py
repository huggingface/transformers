#!/usr/bin/env python3
"""
Example script for exporting Marian models to ONNX format.

This script demonstrates the correct way to export Marian models (including OPUS models)
to ONNX format while maintaining the encoder-decoder architecture and ensuring
correct output quality.

Usage:
    python export_marian_onnx.py --model_name "Helsinki-NLP/opus-mt-en-ar" --output_dir "./onnx_export"
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

from transformers import AutoTokenizer, MarianMTModel


def export_marian_model_to_onnx(model_name: str, output_dir: str, opset_version: int = 17):
    """
    Export a Marian model to ONNX format using the proper encoder-decoder separation.

    Args:
        model_name: Name of the Marian model to export
        output_dir: Directory to save the ONNX models
        opset_version: ONNX opset version to use
    """
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)

    # Set model to evaluation mode
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export encoder
    print("Exporting encoder...")
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    model.export_encoder_to_onnx(encoder_path, opset_version=opset_version, input_shape=(1, 64), device="cpu")
    print(f"Encoder exported to: {encoder_path}")

    # Export decoder
    print("Exporting decoder...")
    decoder_path = os.path.join(output_dir, "decoder.onnx")
    model.export_decoder_to_onnx(decoder_path, opset_version=opset_version, input_shape=(1, 64), device="cpu")
    print(f"Decoder exported to: {decoder_path}")

    return encoder_path, decoder_path


def test_onnx_export(
    model_name: str, encoder_path: str, decoder_path: str, test_sentence: str = "Hello, how are you?"
):
    """
    Test the ONNX export by comparing outputs with the original PyTorch model.

    Args:
        model_name: Name of the original model
        encoder_path: Path to the exported encoder ONNX model
        decoder_path: Path to the exported decoder ONNX model
        test_sentence: Test sentence to translate
    """
    print(f"\nTesting ONNX export with sentence: '{test_sentence}'")

    # Load original model and tokenizer
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Tokenize input
    inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=64)

    # Get PyTorch output
    with torch.no_grad():
        pt_outputs = model.generate(**inputs, max_length=64, num_beams=1, do_sample=False, early_stopping=True)
        pt_translation = tokenizer.decode(pt_outputs[0], skip_special_tokens=True)

    print(f"PyTorch translation: {pt_translation}")

    # Test ONNX models
    try:
        # Load ONNX models
        encoder_session = ort.InferenceSession(encoder_path)
        decoder_session = ort.InferenceSession(decoder_path)

        # Prepare inputs for encoder
        encoder_inputs = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        }

        # Run encoder
        encoder_outputs = encoder_session.run(["hidden_states"], encoder_inputs)
        encoder_hidden_states = encoder_outputs[0]

        # Prepare decoder inputs
        decoder_input_ids = np.full((1, 64), tokenizer.pad_token_id, dtype=np.int64)
        start_id = getattr(model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = tokenizer.pad_token_id
        decoder_input_ids[0, 0] = start_id

        # Run decoder step by step (simulating generation)
        max_length = 64
        generated_tokens = []

        for step in range(max_length - 1):
            decoder_inputs = {
                "input_ids": decoder_input_ids.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
                "encoder_attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
            }

            decoder_outputs = decoder_session.run(["logits"], decoder_inputs)
            logits = decoder_outputs[0]

            # Get next token (greedy decoding)
            next_token_id = np.argmax(logits[0, step, :])

            if next_token_id == tokenizer.eos_token_id:
                break

            if step + 1 < max_length:
                decoder_input_ids[0, step + 1] = next_token_id

            generated_tokens.append(next_token_id)

        # Decode ONNX output
        onnx_translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"ONNX translation: {onnx_translation}")

        # Compare outputs
        if pt_translation.strip() == onnx_translation.strip():
            print("✅ ONNX export successful! Outputs match.")
        else:
            print("⚠️  ONNX export output differs from PyTorch output.")
            print("This might indicate an issue with the export process.")

    except Exception as e:
        print(f"❌ Error testing ONNX models: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export Marian models to ONNX format")
    parser.add_argument(
        "--model_name", type=str, default="Helsinki-NLP/opus-mt-en-ar", help="Name of the Marian model to export"
    )
    parser.add_argument("--output_dir", type=str, default="./onnx_export", help="Directory to save the ONNX models")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version to use")
    parser.add_argument(
        "--test_sentence",
        type=str,
        default="Using handheld GPS devices and programs like Google Earth, members of the Trio Tribe, who live in the rainforests of southern Suriname, map out their ancestral lands to help strengthen their territorial claims.",
        help="Test sentence to translate",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Marian Model ONNX Export Tool")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"ONNX opset version: {args.opset_version}")
    print("=" * 80)

    try:
        # Export model
        encoder_path, decoder_path = export_marian_model_to_onnx(args.model_name, args.output_dir, args.opset_version)

        # Test export
        test_onnx_export(args.model_name, encoder_path, decoder_path, args.test_sentence)

        print("\n✅ Export completed successfully!")
        print(f"Encoder saved to: {encoder_path}")
        print(f"Decoder saved to: {decoder_path}")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
