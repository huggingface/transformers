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
resolution (--text / --file / stdin), output formatting, media loading
(images, audio, video), model loading, and shared CLI option types.
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer


ModelOpt = Annotated[str | None, typer.Option("--model", "-m", help="Model ID or local path.")]
DeviceOpt = Annotated[str | None, typer.Option(help="Device to run on (e.g. 'cpu', 'cuda', 'cuda:0', 'mps').")]
DtypeOpt = Annotated[str, typer.Option(help="Dtype for model weights ('auto', 'float16', 'bfloat16', 'float32').")]
TrustOpt = Annotated[bool, typer.Option(help="Trust remote code from the Hub.")]
TokenOpt = Annotated[str | None, typer.Option(help="HF Hub token for gated/private models.")]
RevisionOpt = Annotated[str | None, typer.Option(help="Model revision (branch, tag, or commit SHA).")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Output results as JSON.")]


def _load_pretrained(model_cls, processor_cls, model_id, device, dtype, trust_remote_code, token, revision):
    """Load a model and its processor/tokenizer with the common CLI options."""
    import torch

    common_kwargs = {}
    if trust_remote_code:
        common_kwargs["trust_remote_code"] = True
    if token:
        common_kwargs["token"] = token
    if revision:
        common_kwargs["revision"] = revision

    model_kwargs = {**common_kwargs}
    if device and device != "cpu":
        model_kwargs["device_map"] = device
    elif device is None:
        model_kwargs["device_map"] = "auto"
    if dtype != "auto":
        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    processor = processor_cls.from_pretrained(model_id, **common_kwargs)
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, processor


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


def load_video(path: str, num_frames: int = 16):
    """
    Load video frames uniformly sampled from a video file.

    Tries ``decord`` first, then falls back to ``av``. Returns a list of
    PIL Images.
    """
    import numpy as np
    from PIL import Image

    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except ImportError:
        pass

    try:
        import av

        container = av.open(path)
        total = container.streams.video[0].frames or 1000
        step = max(1, total // num_frames)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i % step == 0:
                frames.append(frame.to_image())
            if len(frames) >= num_frames:
                break
        container.close()
        return frames
    except ImportError:
        raise SystemExit(
            "Video loading requires 'decord' or 'av'.\nInstall with: pip install decord  (or)  pip install av"
        )


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
