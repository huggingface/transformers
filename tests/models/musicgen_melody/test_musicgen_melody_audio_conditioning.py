# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for MusicgenMelody audio conditioning."""

import math

import numpy as np
import pytest
import torch

from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration


@pytest.mark.parametrize("model_id", ["facebook/musicgen-melody"])
def test_musicgen_melody_audio_conditioning_changes_output(model_id: str):
    """
    Regression test for GH issue #45647: MusicgenMelody ignores audio conditioning.

    Two different reference audios should lead to measurably different generated audio,
    when all other inputs (prompt, seed, generation params) are identical.
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenMelodyForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch_device == "cuda" else torch.float32,
    ).to(torch_device)

    sampling_rate = 32000
    duration = 1.0
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False, dtype=np.float32)

    # Two simple tones with different frequencies
    freq_a = 440.0  # A4
    freq_b = 311.13  # Eb4
    amplitude = 0.4

    ref_a = (amplitude * np.sin(2 * math.pi * freq_a * t)).astype(np.float32)
    ref_b = (amplitude * np.sin(2 * math.pi * freq_b * t)).astype(np.float32)

    # Sanity check: references must actually differ
    assert not np.allclose(ref_a, ref_b), "reference audios are accidentally identical"

    def generate_from_ref(ref_audio: np.ndarray) -> torch.Tensor:
        inputs = processor(
            text=["jazz"],
            audio=ref_audio,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        ).to(torch_device)

        if "input_features" in inputs:
            # Match model dtype to avoid unnecessary casting issues
            inputs["input_features"] = inputs["input_features"].to(model.dtype)

        torch.manual_seed(42)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                guidance_scale=3.0,
            )

        return generated

    gen_a = generate_from_ref(ref_a)
    gen_b = generate_from_ref(ref_b)

    # Compare mean absolute difference between the first codebook outputs.
    # If audio conditioning is ignored, outputs will be (near) identical and diff ~ 0.
    diff = (gen_a[0, 0].float() - gen_b[0, 0].float()).abs().mean().item()

    assert diff > 1e-3, f"audio conditioning appears inactive, mean abs diff={diff:.8f}"
