# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Shared helpers used by all agentic CLI commands.

These are internal utilities — not CLI commands themselves. They handle input
resolution (--text / --file / stdin), output formatting, and media loading
(images, audio) so that each command stays thin.
"""

import json
import sys
from pathlib import Path
from typing import Any


def resolve_input(text: str | None = None, file: str | None = None) -> str:
    """
    Return text from one of three sources, in priority order:

    1. ``--text "..."`` — inline string
    2. ``--file path`` — read from a file
    3. stdin — piped input (e.g. ``echo "hello" | transformers classify``)

    Raises ``SystemExit`` if none of the three are provided.
    """
    if text is not None:
        return text
    if file is not None:
        return Path(file).read_text()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Error: provide --text, --file, or pipe input via stdin.")


def format_output(result: Any, output_json: bool = False) -> str:
    """
    Format pipeline output for display.

    When ``output_json=True``, returns a JSON string (useful for agents that
    need to parse results programmatically). Otherwise, returns a
    human-readable multi-line string.
    """
    if output_json:
        return json.dumps(result, indent=2, default=str)

    if isinstance(result, list):
        lines = []
        for item in result:
            if isinstance(item, dict):
                lines.append("  ".join(f"{k}: {v}" for k, v in item.items()))
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        lines.append("  ".join(f"{k}: {v}" for k, v in sub.items()))
                    else:
                        lines.append(str(sub))
            else:
                lines.append(str(item))
        return "\n".join(lines)

    if isinstance(result, dict):
        return "\n".join(f"{k}: {v}" for k, v in result.items())

    return str(result)


def load_image(path: str):
    """
    Load an image from a local file path or a URL.

    Returns a PIL Image. Requires ``Pillow`` (``pip install Pillow``).
    For URLs, also requires ``requests``.
    """
    from PIL import Image

    if path.startswith("http://") or path.startswith("https://"):
        import requests

        return Image.open(requests.get(path, stream=True).raw)
    return Image.open(path)


def load_audio(path: str, sampling_rate: int = 16000):
    """
    Load an audio file, resampling to ``sampling_rate`` Hz.

    Tries ``librosa`` first (supports resampling). Falls back to
    ``soundfile`` if librosa is not installed, but will error if the
    file's sample rate doesn't match the target.
    """
    import numpy as np

    try:
        import librosa

        audio, _ = librosa.load(path, sr=sampling_rate)
        return audio
    except ImportError:
        import soundfile as sf

        audio, sr = sf.read(path)
        if sr != sampling_rate:
            raise SystemExit(
                f"Audio sample rate is {sr} but model expects {sampling_rate}. "
                "Install librosa (`pip install librosa`) for automatic resampling."
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)
