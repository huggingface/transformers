# Copyright 2025 The HuggingFace Inc. team.
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
Tests comparing the new AudioProcessor classes against the legacy FeatureExtractor classes
for Wav2Vec2 and Whisper only.

For each model, we:
1. Instantiate the FeatureExtractor via from_pretrained (from the Hub)
2. Instantiate the corresponding AudioProcessor directly
3. Run both on the same batched audio input
4. Assert torch.equal on the main output tensors
"""


import numpy as np
import pytest
import torch

from transformers.models.wav2vec2.audio_processing_wav2vec2 import Wav2Vec2AudioProcessor, Wav2Vec2FeatureExtractor
from transformers.models.whisper.audio_processing_whisper import WhisperAudioProcessor, WhisperFeatureExtractor


MODEL_CONFIGS = [
    {
        "name": "wav2vec2",
        "hub_repo": "facebook/wav2vec2-large-960h-lv60-self",
        "fe_class": Wav2Vec2FeatureExtractor,
        "ap_class": Wav2Vec2AudioProcessor,
        "fe_output_key": "input_values",
        "sample_rate": 16000,
    },
    {
        "name": "whisper",
        "hub_repo": "openai/whisper-small",
        "fe_class": WhisperFeatureExtractor,
        "ap_class": WhisperAudioProcessor,
        "fe_output_key": "input_features",
        "sample_rate": 16000,
        "ap_kwargs": {"max_length": None, "truncation": False},
    },
]


def _make_audio_batch(sample_rate: int, seed: int = 42) -> list[np.ndarray]:
    """Create a deterministic batched audio input: two clips of different lengths."""
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal(sample_rate).astype(np.float32),  # 1 second
        rng.standard_normal(sample_rate * 2).astype(np.float32),  # 2 seconds
    ]


@pytest.mark.parametrize(
    "config",
    MODEL_CONFIGS,
    ids=[c["name"] for c in MODEL_CONFIGS],
)
def test_audio_processor_matches_feature_extractor(config):
    hub_repo = config["hub_repo"]
    fe_class = config["fe_class"]
    ap_class = config["ap_class"]
    fe_output_key = config["fe_output_key"]
    sample_rate = config["sample_rate"]

    fe = fe_class.from_pretrained(hub_repo)
    ap = ap_class()

    audio_batch = _make_audio_batch(sample_rate)

    default_kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
        "padding": True,
    }

    fe_kwargs = {**default_kwargs, **config.get("fe_kwargs", {})}
    ap_kwargs = {**default_kwargs, **config.get("ap_kwargs", {})}

    fe_output = fe(audio_batch, **fe_kwargs)
    ap_output = ap(audio_batch, **ap_kwargs)

    # Map feature extractor keys to audio processor keys if needed
    fe_to_ap_key_map = {
        "input_features": "audio_features",
        "input_values": "audio_values",
    }

    for fe_key in fe_output.keys():
        if fe_key == "attention_mask":
            continue
        ap_key = fe_to_ap_key_map.get(fe_key, fe_key)
        assert ap_key in ap_output, f"Key {ap_key} (from FE key {fe_key}) not found in audio processor output"
        fe_tensor = fe_output[fe_key]
        ap_tensor = ap_output[ap_key]

        if not isinstance(fe_tensor, torch.Tensor):
            fe_tensor = torch.tensor(fe_tensor)
        if not isinstance(ap_tensor, torch.Tensor):
            ap_tensor = torch.tensor(ap_tensor)

        assert fe_tensor.shape == ap_tensor.shape, (
            f"Shape mismatch for key '{fe_key}' (ap key '{ap_key}'): fe {fe_tensor.shape} vs ap {ap_tensor.shape}"
        )
        # Note: We now use torch.equal to require exact equality for tensors.
        assert torch.equal(fe_tensor, ap_tensor), (
            f"Value mismatch for key '{fe_key}' (ap key '{ap_key}'): tensors are not exactly equal"
        )

if __name__ == "__main__":
    test_audio_processor_matches_feature_extractor(MODEL_CONFIGS[0])
