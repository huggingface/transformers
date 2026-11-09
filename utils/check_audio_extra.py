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
"""
Smoke test for an environment where only `pip install transformers[audio]` was run (see `extras-smoke-test.yml`).

Checks that the core audio path works with exactly what `[audio]` installs, and that the dependencies moved out of it
did not creep back in.

Usage: python utils/check_audio_extra.py
"""

import importlib.util
import math
import os
import tempfile

import numpy as np
import soundfile as sf
from packaging import version

from transformers.audio_utils import load_audio
from transformers.utils import is_torchcodec_available


# Not part of `[audio]` anymore: they are model-specific, deprecated or test-only.
MOVED_OUT = ["librosa", "pyctcdecode", "kenlm", "phonemizer"]


def main():
    leaked = [name for name in MOVED_OUT if importlib.util.find_spec(name) is not None]
    assert not leaked, f"`[audio]` should not install {leaked}"
    assert version.parse(np.__version__).major >= 2, f"`[audio]` downgraded numpy to {np.__version__}"
    assert not is_torchcodec_available(), "torchcodec is not part of `[audio]`, `auto` must resolve to soundfile"

    src_sampling_rate, sampling_rate, num_samples = 44100, 16000, 44100 + 17
    stereo = (np.random.default_rng(0).standard_normal((num_samples, 2)) * 0.1).astype(np.float32)
    expected_length = math.ceil(num_samples * sampling_rate / src_sampling_rate)

    with tempfile.TemporaryDirectory() as tmpdir:
        for audio_format in ["WAV", "FLAC", "MP3"]:
            path = os.path.join(tmpdir, f"sample.{audio_format.lower()}")
            sf.write(path, stereo, src_sampling_rate, format=audio_format)
            audio = load_audio(path, sampling_rate=sampling_rate)
            assert audio.ndim == 1 and audio.dtype == np.float32, f"{audio_format}: {audio.shape}, {audio.dtype}"
            assert len(audio) == expected_length, f"{audio_format}: {len(audio)} samples, expected {expected_length}"
            print(f"{audio_format}: OK ({len(audio)} samples at {sampling_rate} Hz)")


if __name__ == "__main__":
    main()
