# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import numpy as np
import pytest

from transformers import DacFeatureExtractor
from transformers.models.voxcpm2.processing_voxcpm2 import VoxCPM2Processor
from transformers.testing_utils import require_torch

from .test_tokenization_voxcpm2 import get_tiny_voxcpm2_tokenizer


def get_tiny_voxcpm2_processor() -> VoxCPM2Processor:
    tokenizer = get_tiny_voxcpm2_tokenizer()
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|audio_start|>",
                "<|audio_prompt_start|>",
                "<|audio_prompt_end|>",
            ]
        }
    )
    feature_extractor = DacFeatureExtractor(sampling_rate=16000, hop_length=1, return_attention_mask=True)
    return VoxCPM2Processor(feature_extractor, tokenizer, audio_patch_size=4)


@require_torch
def test_processor_packs_all_generation_modes():
    processor = get_tiny_voxcpm2_processor()
    prompt_audio = np.arange(5, dtype=np.float32)
    reference_audio = np.arange(3, dtype=np.float32)
    target_ids = processor.tokenizer("A", add_special_tokens=False).input_ids
    continuation_ids = processor.tokenizer("BA", add_special_tokens=False).input_ids

    zero_shot = processor(text="A", return_tensors="pt")
    continuation = processor(
        text="A",
        audio=prompt_audio,
        prompt_text="B",
        sampling_rate=16000,
        return_tensors="pt",
    )
    reference = processor(
        text="A",
        reference_audio=reference_audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    combined = processor(
        text="A",
        audio=prompt_audio,
        prompt_text="B",
        reference_audio=reference_audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    audio_start = processor.audio_start_token_id
    reference_start = processor.reference_audio_start_token_id
    reference_end = processor.reference_audio_end_token_id
    assert zero_shot.input_ids.tolist() == [target_ids + [audio_start]]
    assert continuation.input_ids.tolist() == [continuation_ids + [audio_start, 0, 0]]
    assert reference.input_ids.tolist() == [[reference_start, 0, reference_end] + target_ids + [audio_start]]
    assert combined.input_ids.tolist() == [
        [reference_start, 0, reference_end] + continuation_ids + [audio_start, 0, 0]
    ]

    for model_inputs in (zero_shot, continuation, reference, combined):
        assert model_inputs.attention_mask.bool().all()
        assert (model_inputs.text_mask + model_inputs.audio_mask == 1).all()


@require_torch
def test_processor_preserves_audio_lengths_and_validates_inputs():
    processor = get_tiny_voxcpm2_processor()
    processor.feature_extractor.hop_length = 4
    prompt_audio = np.arange(5, dtype=np.float32)
    reference_audio = np.arange(3, dtype=np.float32)

    model_inputs = processor(
        text="A",
        audio=prompt_audio,
        prompt_text="B",
        reference_audio=reference_audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    assert model_inputs.prompt_input_values.shape == (1, 1, 8)
    assert model_inputs.prompt_attention_mask.tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    assert model_inputs.reference_input_values.shape == (1, 1, 4)
    assert model_inputs.reference_attention_mask.tolist() == [[1, 1, 1, 0]]
    assert model_inputs.audio_mask.sum() == 3

    with pytest.raises(ValueError, match="provided together"):
        processor(text="A", audio=prompt_audio, sampling_rate=16000)
    with pytest.raises(ValueError, match="provided together"):
        processor(text="A", prompt_text="B")
    with pytest.raises(ValueError, match="one audio sample"):
        processor(
            text="A",
            audio=[prompt_audio, prompt_audio],
            prompt_text="B",
            sampling_rate=16000,
        )
    with pytest.raises(ValueError, match="Expected mono audio"):
        processor(
            text="A",
            audio=np.zeros((2, 5), dtype=np.float32),
            prompt_text="B",
            sampling_rate=16000,
        )
