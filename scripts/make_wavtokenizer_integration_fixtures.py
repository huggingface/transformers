# Copyright 2026 The SwissAI Initiative and The HuggingFace Inc. team. All rights reserved.
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
"""Reproducer for the expected values of the WavTokenizer integration tests in transformers, stored in
`tests/fixtures/wavtokenizer/expected_results.json`.

Runs the original implementation (https://github.com/jishengpeng/WavTokenizer, tested at commit `5cf440d`) with the
released `wavtokenizer_large_unify_600_24k.ckpt` on two speech clips from `hf-internal-testing/dummy-audio-samples`
(native 24 kHz, so no resampling is involved) and writes the codes, the first decoded samples and the codec error.

Both clips are right-padded with zeros to a common length, as the feature extractor does, and encoded as one batch.
The original does not mask the padding, so the trailing codes of the shorter clip depend on it; the expected codes of
each row are the codes covering its valid samples. Row 0 is the longest clip, so it is not padded and also serves as
the reference for encoding it on its own.

Requirements: a clone of the original repository, `torch`, `torchaudio`, `einops`, `pyyaml` and `scipy` (for the
original code), plus `numpy`, `soundfile` and `huggingface_hub`.

    python make_wavtokenizer_integration_fixtures.py --original_repo WavTokenizer --output expected_results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download


CHECKPOINT = ("novateur/WavTokenizer-large-unify-40token", "wavtokenizer_large_unify_600_24k.ckpt")
CONFIG = "configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
AUDIO_FILES = ["en-Carter_man.wav", "en-Frank_man.wav"]
SAMPLING_RATE = 24000
HOP_LENGTH = 600
NUM_AUDIO_VALUES = 50


def compute_rmse(arr1, arr2):
    """Same as `tests/utils/test_audio_utils.py::compute_rmse`: RMSE after L2 normalization, on the shorter length."""
    arr1, arr2 = np.asarray(arr1).squeeze(), np.asarray(arr2).squeeze()
    length = min(arr1.shape[-1], arr2.shape[-1])
    arr1, arr2 = arr1[..., :length], arr2[..., :length]
    return float(np.sqrt(((arr1 / np.linalg.norm(arr1) - arr2 / np.linalg.norm(arr2)) ** 2).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original_repo", required=True, help="Path to a clone of jishengpeng/WavTokenizer")
    parser.add_argument("--output", required=True, help="Where to write the expected values (JSON)")
    parser.add_argument("--device", default="cpu", help="Device to run the original model on")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.original_repo).resolve()))
    from decoder.pretrained import WavTokenizer  # noqa: PLC0415

    model = WavTokenizer.from_pretrained0802(str(Path(args.original_repo) / CONFIG), hf_hub_download(*CHECKPOINT))
    model = model.to(args.device).eval()
    bandwidth_id = torch.tensor([0], device=args.device)

    def encode(waveforms):  # (batch, samples) -> (batch, codes)
        _, codes = model.encode_infer(waveforms.to(args.device), bandwidth_id=bandwidth_id)
        return codes.reshape(waveforms.shape[0], -1)

    def decode(codes):  # (batch, codes) -> (batch, samples)
        return model.decode(model.codes_to_features(codes.unsqueeze(0)), bandwidth_id=bandwidth_id).cpu()

    audios = []
    for filename in AUDIO_FILES:
        path = hf_hub_download("hf-internal-testing/dummy-audio-samples", filename, repo_type="dataset")
        audio, sampling_rate = sf.read(path, dtype="float32")
        assert sampling_rate == SAMPLING_RATE and audio.ndim == 1
        audios.append(audio)

    with torch.no_grad():
        lengths = [len(audio) for audio in audios]
        batch = torch.zeros(len(audios), max(lengths))
        for row, audio in enumerate(audios):
            batch[row, : len(audio)] = torch.from_numpy(audio)
        batch_codes = encode(batch)
        batch_audio = decode(batch_codes)

    valid_codes = [math.ceil(length / HOP_LENGTH) for length in lengths]
    expected = {
        "audio_codes": [batch_codes[row, :num_codes].tolist() for row, num_codes in enumerate(valid_codes)],
        "audio_values": [batch_audio[row, :NUM_AUDIO_VALUES].tolist() for row in range(len(audios))],
        "codec_errors": [
            compute_rmse(batch[row, :length].numpy(), batch_audio[row, :length].numpy())
            for row, length in enumerate(lengths)
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(expected, file)
        file.write("\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
