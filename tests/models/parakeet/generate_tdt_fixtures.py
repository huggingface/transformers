#!/usr/bin/env python3
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
Script to generate expected test fixtures for ParakeetForTDT integration tests.

This script runs the TDT model on LibriSpeech samples and saves the outputs
to JSON fixtures that are used by the integration tests.

Usage:
    python tests/models/parakeet/generate_tdt_fixtures.py

Requirements:
    - torch
    - transformers
    - datasets
"""

import json
from pathlib import Path

import torch
from datasets import Audio, load_dataset

from transformers import AutoProcessor, ParakeetForTDT


def main():
    # TODO: Change to "nvidia/parakeet-tdt-0.6b-v3" once NVIDIA adds HF format to their repo
    checkpoint_name = "MaksL/parakeet-tdt-0.6b-v3"
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {checkpoint_name}...")
    processor = AutoProcessor.from_pretrained(checkpoint_name)
    model = ParakeetForTDT.from_pretrained(checkpoint_name, torch_dtype=dtype, device_map=device)
    model.eval()

    print("Loading dataset...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

    # Sort by ID to ensure reproducibility
    speech_samples = dataset.sort("id")

    fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "parakeet"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    # Generate single sample fixture
    print("Generating single sample fixture...")
    single_sample = [speech_samples[0]["audio"]["array"]]
    inputs = processor(single_sample)
    inputs.to(device, dtype=dtype)

    with torch.no_grad():
        output = model.generate(**inputs, return_dict_in_generate=True, return_timestamps=True)

    single_fixture = {
        "transcriptions": processor.batch_decode(output.sequences, skip_special_tokens=True),
        "token_ids": output.sequences.cpu().tolist(),
    }

    single_path = fixtures_dir / "expected_results_tdt_single.json"
    with open(single_path, "w") as f:
        json.dump(single_fixture, f)
    print(f"Saved: {single_path}")

    # Generate batch fixture (5 samples)
    print("Generating batch fixture...")
    batch_samples = [speech_samples[i]["audio"]["array"] for i in range(5)]
    inputs = processor(batch_samples)
    inputs.to(device, dtype=dtype)

    with torch.no_grad():
        output = model.generate(**inputs, return_dict_in_generate=True, return_timestamps=True)

    batch_fixture = {
        "transcriptions": processor.batch_decode(output.sequences, skip_special_tokens=True),
        "token_ids": output.sequences.cpu().tolist(),
    }

    batch_path = fixtures_dir / "expected_results_tdt_batch.json"
    with open(batch_path, "w") as f:
        json.dump(batch_fixture, f)
    print(f"Saved: {batch_path}")

    print("\nFixtures generated successfully!")
    print(f"\nSingle sample transcription:\n  {single_fixture['transcriptions'][0]}")
    print("\nBatch transcriptions:")
    for i, t in enumerate(batch_fixture["transcriptions"]):
        print(f"  [{i}] {t}")


if __name__ == "__main__":
    main()
