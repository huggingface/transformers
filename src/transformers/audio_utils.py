# Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors.
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
Audio processing functions to extract features from audio waveforms. This code is pure numpy to support all frameworks
and remove unnecessary dependencies.
"""

import base64
import importlib
import io
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union
from urllib.parse import urlparse

import httpx
import numpy as np
from packaging import version

from .utils import (
    is_librosa_available,
    is_numpy_array,
    is_soundfile_available,
    is_torch_available,
    is_torch_tensor,
    is_torchaudio_available,
    is_torchcodec_available,
    requires_backends,
)
from .utils.generic import retry


if TYPE_CHECKING:
    import torch

if is_soundfile_available():
    import soundfile as sf

if is_librosa_available():
    import librosa

    # TODO: @eustlb, we actually don't need librosa but soxr is installed with librosa
    import soxr

if is_torchaudio_available():
    import torchaudio

if is_torchcodec_available():
    TORCHCODEC_VERSION = version.parse(importlib.metadata.version("torchcodec"))

AudioInput = Union[np.ndarray, "torch.Tensor", Sequence[np.ndarray], Sequence["torch.Tensor"]]


@dataclass(frozen=True)
class StftConfig:
    n_fft: int = 400
    win_length: int | None = None
    hop_length: int | None = None
    window_fn: str = "hann_window"
    wkwargs: dict | None = None
    power: float = 2.0
    center: bool = True
    pad_mode: str = "reflect"
    normalized: bool = False
    onesided: bool | None = None
    periodic: bool = True
    left_align_fft: bool = False
    window_dtype: str | None = None

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "StftConfig":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass(frozen=True)
class MelScaleConfig:
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = None
    mel_scale: str = "htk"
    norm: str | None = None
    triangularize_in_mel_space: bool = False
    frequency_bin_mode: str = "rfft"
    computation_dtype: str | None = None
    bands_to_zero: int = 0
    matmul_order: str = "filters_first"

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "MelScaleConfig":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass(frozen=True)
class SpectrogramConfig:
    stft_config: StftConfig = field(default_factory=StftConfig)
    mel_scale_config: MelScaleConfig | None = None
    log_mode: str = "log10"
    chunk_length: int | None = None
    preemphasis: float | None = None
    # Where preemphasis is applied: "per_frame" (default; on each framed window, first sample
    # scaled by 1-p) or "waveform" (on the raw waveform before framing, first sample unchanged,
    # padded samples zeroed via audio_ranges). ASR models (Parakeet/Cohere/Nemotron) use "waveform".
    preemphasis_mode: str = "per_frame"
    remove_dc_offset: bool = False
    mel_floor: float = 1e-10
    waveform_scale: float | None = None
    computation_dtype: str | None = None
    skip_last_frame: bool = False
    clip_max_offset: float | None = None
    post_log_shift: float | None = None
    post_log_scale: float | None = None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in SpectrogramConfig.")

    def __iter__(self):
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                if hasattr(val, "to_dict"):
                    yield f.name, val.to_dict()
                else:
                    yield f.name, val

    def __eq__(self, other):
        if isinstance(other, dict):
            return dict(self) == other
        if isinstance(other, SpectrogramConfig):
            return tuple(getattr(self, f.name) for f in fields(self)) == tuple(
                getattr(other, f.name) for f in fields(self)
            )
        return NotImplemented

    def to_dict(self) -> dict:
        return dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpectrogramConfig":
        kwargs = {k: v for k, v in d.items() if k in {f.name for f in fields(cls)}}
        if "stft_config" in kwargs and isinstance(kwargs["stft_config"], dict):
            kwargs["stft_config"] = StftConfig.from_dict(kwargs["stft_config"])
        if "mel_scale_config" in kwargs and isinstance(kwargs["mel_scale_config"], dict):
            kwargs["mel_scale_config"] = MelScaleConfig.from_dict(kwargs["mel_scale_config"])
        return cls(**kwargs)


@retry(exceptions=(httpx.HTTPError,))
def _fetch_audio_bytes(url: str, timeout: float | None = 10.0) -> bytes:
    """Fetch audio bytes from a URL with automatic retry and exponential backoff."""
    response = httpx.get(url, follow_redirects=True, timeout=timeout)
    response.raise_for_status()
    return response.content


_NEEDS_TORCHCODEC = "Install torchcodec>=0.3.0 (`pip install torchcodec`) to load audio from this source."


TORCHCODEC_ONLY_FILETYPES = frozenset(
    {
        "3gp",
        "aac",
        "ac3",
        "amr",
        "avi",
        "flv",
        "m4a",
        "m4v",
        "mkv",
        "mov",
        "mp4",
        "mpg",
        "ogv",
        "sox",
        "ts",
        "webm",
        "wma",
        "wmv",
        "wv",
    }
)


def _format_from_source(audio: str) -> "str | None":
    """Best-effort format token from the source *string* — the file extension (paths and URLs) or
    the media subtype (`data:` URIs) — without resolving or decoding it. Returns None when the
    string carries no hint, e.g. a raw base64 payload."""
    if audio.startswith("data:"):
        media_type = audio[len("data:") :].split(",", 1)[0].split(";", 1)[0]
        return media_type.rpartition("/")[2].removeprefix("x-") or None
    path = urlparse(audio).path if audio.startswith(("http://", "https://")) else audio
    return os.path.splitext(path)[1].lstrip(".").lower() or None


def get_audio_filetype(data: bytes) -> str:
    """Identify a file's container/codec from its magic bytes.

    A few extensions are byte-identical in their headers and collapse to a canonical type:
    ``wavex`` -> ``wav`` and ``m4v``/``hevc.mp4`` -> ``mp4`` (all carry the ``isom`` ftyp brand).

    Raises ValueError if the bytes match no supported filetype.
    """
    head = data[:64]

    # Containers that host several filetypes -> sniff a bit deeper.
    if head[4:8] == b"ftyp":  # ISO-BMFF: m4v & hevc share the 'isom' brand -> mp4
        brand = head[8:12]
        return (
            "3gp" if brand[:3] == b"3gp" else "m4a" if brand[:3] == b"M4A" else "mov" if brand[:2] == b"qt" else "mp4"
        )
    if head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI "):
        return "wav" if head[8:12] == b"WAVE" else "avi"
    if head[:4] == b"riff" and head[4:8] == bytes.fromhex("2e91cf11"):  # Wave64
        return "w64"
    if head[:4] == bytes.fromhex("1a45dfa3"):  # EBML: Matroska vs WebM
        return "webm" if b"webm" in head else "mkv"
    if head[:4] == b"OggS":  # OGG: Opus / Theora (ogv) / Vorbis (ogg)
        page = data[:128]
        return "opus" if b"OpusHead" in page else "ogv" if b"theora" in page else "ogg"
    if head[:16] == bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c"):  # ASF
        return "wmv" if bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b") in data else "wma"
    if head[:1] == b"\xff" and len(head) > 1 and head[1] & 0xE0 == 0xE0:  # MPEG/AAC sync
        if head[1] & 0xF6 == 0xF0:  # ADTS layer bits 00 -> AAC
            return "aac"
        layer = head[1] >> 1 & 0x3  # MPEG audio layer field (II -> mp2, III -> mp3)
        if layer in (0b10, 0b01):
            return "mp2" if layer == 0b10 else "mp3"
    if head[:1] == b"\x47" and len(data) > 188 and data[188] == 0x47:
        return "ts"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"

    # Single fixed-signature formats, keyed by their leading bytes.
    signatures = {
        b"fLaC": "flac",
        b"RF64": "rf64",
        b"caff": "caf",
        b".snd": "au",
        b"#!AMR": "amr",
        b"wvpk": "wv",
        b".SoX": "sox",
        b"XoS.": "sox",
        b"Creative Voice File": "voc",
        b"\x64\xa3\x01\x00": "sf",
        b"\x00\x01\xa3\x64": "sf",
        b"\x0b\x77": "ac3",
        b"\x00\x00\x01\xba": "mpg",
        b"FLV": "flv",
        b"ID3": "mp3",
    }
    for sig, filetype in signatures.items():
        if head.startswith(sig):
            return filetype

    raise ValueError("not supported filetype")


def _resolve_audio_source(audio: str, timeout: float | None = None) -> "str | bytes":
    """Resolve an audio source string to a local file path or raw bytes for a decoder.

    Accepts `http(s)://` URLs (fetched with retry), local file paths (returned unchanged),
    and base64 strings (optionally wrapped as a `data:...` URI).
    """
    if audio.startswith(("http://", "https://")):
        return _fetch_audio_bytes(audio, timeout=timeout)
    if os.path.isfile(audio):
        return audio
    # Not a URL or a local path — assume base64, optionally wrapped as a `data:<media-type>;base64,` URI
    if audio.startswith("data:"):
        audio = audio.split(",", 1)[1]
    try:
        return base64.b64decode(audio)
    except Exception as e:
        raise ValueError(
            "Incorrect audio source. Must be a valid URL starting with `http://` or `https://`, "
            f"a valid path to an audio file, or a base64 encoded string. Got {audio}. Failed with {e}"
        )


def load_audio(audio: str | np.ndarray, sampling_rate=16000, timeout=None, backend: str = "auto") -> np.ndarray:
    """
    Loads `audio` to an np.ndarray object.

    Args:
        audio (`str` or `np.ndarray`):
            The audio to be loaded to the numpy array format. If a `str`, it can be an `http(s)://`
            URL, a local file path, or a base64-encoded string (optionally wrapped as a
            `data:<media-type>;base64,` URI).
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate to be used when loading the audio. It should be same as the
            sampling rate the model you will be using further was trained with.
        timeout (`float`, *optional*):
            The timeout value in seconds for the URL request.
        backend (`str`, *optional*, defaults to `"auto"`):
            Decoding backend: `"auto"` uses torchcodec when available (>=0.3.0) and falls back to
            librosa; `"torchcodec"`, `"librosa"` or `"torchaudio"` force that backend (and error if it
            is missing). `"torchaudio"` decodes with `torchaudio.load` and resamples with
            `torchaudio.functional.resample` (matches serving stacks such as sglang bit-for-bit).

    Returns:
        `np.ndarray`: A numpy array representing the audio.
    """
    if isinstance(audio, np.ndarray):
        return audio
    if not isinstance(audio, str):
        raise TypeError(
            "Incorrect format used for `audio`. Should be a numpy array or a `str`: an `http(s)://` URL, "
            "a local file path, or a base64-encoded string (optionally wrapped as a `data:...` URI)."
        )

    # torchcodec handles audio/video; librosa only plain audio. `backend` lets callers pin one.
    if backend == "auto":
        resolved_backend = (
            "torchcodec" if is_torchcodec_available() and version.parse("0.3.0") <= TORCHCODEC_VERSION else "librosa"
        )
    elif backend in ("torchcodec", "librosa", "torchaudio"):
        resolved_backend = backend
    else:
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'torchcodec', 'librosa', or 'torchaudio'.")
    # soundfile-based backends (librosa / torchaudio) cannot decode the video-ish formats below.
    use_torchcodec = resolved_backend == "torchcodec"

    # 1. Identify the format from the source string (extension / `data:` media type), without fetching.
    filetype = _format_from_source(audio)
    # 2. With librosa as the only backend, fail fast and clearly on a format it cannot decode.
    if not use_torchcodec and filetype in TORCHCODEC_ONLY_FILETYPES:
        raise RuntimeError(
            f"The audio source is a '{filetype}' file, which librosa cannot decode. {_NEEDS_TORCHCODEC}"
        )

    # 3. Resolve to local path or bytes; sniff format for raw base64 payloads before passing to librosa.
    source = _resolve_audio_source(audio, timeout=timeout)
    if not use_torchcodec and filetype is None and isinstance(source, bytes):
        try:
            filetype = get_audio_filetype(source)
        except ValueError:
            filetype = None
        if filetype in TORCHCODEC_ONLY_FILETYPES:
            raise RuntimeError(
                f"The audio source is a '{filetype}' file, which librosa cannot decode. {_NEEDS_TORCHCODEC}"
            )

    # 4. Decode with the selected backend (`requires_backends` raises a clear error if it is missing).
    if use_torchcodec:
        requires_backends(load_audio, ["torchcodec"])
        from torchcodec.decoders import AudioDecoder

        # `num_channels=1` matches what most models expect and librosa's default.
        return AudioDecoder(source, sample_rate=sampling_rate, num_channels=1).get_all_samples().data[0].numpy()

    if resolved_backend == "torchaudio":
        requires_backends(load_audio, ["torchaudio"])
        waveform, src_sampling_rate = torchaudio.load(BytesIO(source) if isinstance(source, bytes) else source)
        waveform = waveform.mean(dim=0)  # to mono

        if src_sampling_rate != sampling_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=src_sampling_rate, new_freq=sampling_rate)
        return waveform.numpy().astype(np.float32)

    requires_backends(load_audio, ["librosa"])
    return librosa.load(BytesIO(source) if isinstance(source, bytes) else source, sr=sampling_rate)[0]


def load_audio_torchcodec(audio: str | np.ndarray, sampling_rate=16000, timeout=None) -> np.ndarray:
    """Deprecated. Use [`load_audio`] instead (equivalent to `backend="torchcodec"`)."""
    warnings.warn(
        "`load_audio_torchcodec` is deprecated and will be removed in a future version. "
        'Use `load_audio(..., backend="torchcodec")` instead.',
        FutureWarning,
    )
    return load_audio(audio, sampling_rate=sampling_rate, timeout=timeout, backend="torchcodec")


def load_audio_librosa(audio: str | np.ndarray, sampling_rate=16000, timeout=None) -> np.ndarray:
    """Deprecated. Use [`load_audio`] instead (equivalent to `backend="librosa"`)."""
    warnings.warn(
        "`load_audio_librosa` is deprecated and will be removed in a future version. "
        'Use `load_audio(..., backend="librosa")` instead.',
        FutureWarning,
    )
    return load_audio(audio, sampling_rate=sampling_rate, timeout=timeout, backend="librosa")


def load_audio_as(
    audio: str,
    return_format: str,
    timeout: int | None = None,
    force_mono: bool = False,
    sampling_rate: int | None = None,
) -> str | dict[str, Any] | io.BytesIO | None:
    """
    Load audio from either a local file path or URL and return in specified format.

    Args:
        audio (`str`): Either a local file path or a URL to an audio file
        return_format (`str`): Format to return the audio in:
            - "base64": Base64 encoded string
            - "dict": Dictionary with data and format
            - "buffer": BytesIO object
        timeout (`int`, *optional*): Timeout for URL requests in seconds
        force_mono (`bool`): Whether to convert stereo audio to mono
        sampling_rate (`int`, *optional*): If provided, the audio will be resampled to the specified sampling rate.

    Returns:
        `Union[str, Dict[str, Any], io.BytesIO, None]`:
            - `str`: Base64 encoded audio data (if return_format="base64")
            - `dict`: Dictionary with 'data' (base64 encoded audio data) and 'format' keys (if return_format="dict")
            - `io.BytesIO`: BytesIO object containing audio data (if return_format="buffer")
    """
    requires_backends(load_audio_as, ["librosa"])

    if return_format not in ["base64", "dict", "buffer"]:
        raise ValueError(f"Invalid return_format: {return_format}. Must be 'base64', 'dict', or 'buffer'")

    try:
        # Load audio bytes from URL or file
        audio_bytes = None
        if audio.startswith(("http://", "https://")):
            audio_bytes = _fetch_audio_bytes(audio, timeout=timeout)
        elif os.path.isfile(audio):
            with open(audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        else:
            raise ValueError(f"File not found: {audio}")

        # Process audio data
        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as f:
                audio_array = f.read(dtype="float32")
                original_sr = f.samplerate
                audio_format = f.format
                if sampling_rate is not None and sampling_rate != original_sr:
                    # Resample audio to target sampling rate
                    audio_array = soxr.resample(audio_array, original_sr, sampling_rate, quality="HQ")
                else:
                    sampling_rate = original_sr

        # Convert to mono if needed
        if force_mono and audio_array.ndim != 1:
            audio_array = audio_array.mean(axis=1)

        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format=audio_format.upper())
        buffer.seek(0)

        if return_format == "buffer":
            return buffer
        elif return_format == "base64":
            return base64.b64encode(buffer.read()).decode("utf-8")
        elif return_format == "dict":
            return {
                "data": base64.b64encode(buffer.read()).decode("utf-8"),
                "format": audio_format.lower(),
            }

    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")


def conv1d_output_length(module: "torch.nn.Conv1d", input_length: int) -> int:
    """
    Computes the output length of a 1D convolution layer according to torch's documentation:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    """
    return int(
        (input_length + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
        / module.stride[0]
        + 1
    )


def is_valid_audio(audio):
    return (
        is_numpy_array(audio)
        or is_torch_tensor(audio)
        or (isinstance(audio, (list, tuple)) and isinstance(audio[0], float))
    )


def is_valid_list_of_audio(audio):
    return audio and all(is_valid_audio(audio_i) for audio_i in audio)


def make_list_of_audio(
    audio: list[AudioInput] | AudioInput,
) -> AudioInput:
    """
    Ensure that the output is a list of audio.
    Args:
        audio (`Union[list[AudioInput], AudioInput]`):
            The input audio.
    Returns:
        list: A list of audio.
    """
    # If it's a list of audios, it's already in the right format
    if isinstance(audio, (list, tuple)) and is_valid_list_of_audio(audio):
        return audio

    # If it's a single audio, convert it to a list of
    if is_valid_audio(audio):
        return [audio]

    raise ValueError("Invalid input type. Must be a single audio or a list of audio")


def make_list_of_audio_chat_template(
    audio: list[AudioInput] | AudioInput | str | list[str],
) -> AudioInput:
    """
    Ensure that the output is a list of audio. Unlike `make_list_of_audio`, this function also accepts a URL string or
    local path, as accepted by chat templates.

    Args:
        audio (`Union[list[AudioInput], AudioInput]`):
            The input audio. Can be a URL string, local path, numpy/torch array,  or a list of these.
    Returns:
        list: A list of audio.
    """

    # Handle string inputs
    if isinstance(audio, str):
        return [audio]
    if isinstance(audio, (list, tuple)) and audio and all(isinstance(a, str) for a in audio):
        return list(audio)

    # Handle numpy/torch array inputs
    return make_list_of_audio(audio)


def hertz_to_octave(freq: float | np.ndarray, tuning: float = 0.0, bins_per_octave: int = 12):
    """
    Convert frequency from hertz to fractional octave numbers.
    Adapted from *librosa*.

    Args:
        freq (`float` or `np.ndarray`):
            The frequency, or multiple frequencies, in hertz (Hz).
        tuning (`float`, defaults to `0.`):
            Tuning deviation from the Stuttgart pitch (A440) in (fractional) bins per octave.
        bins_per_octave (`int`, defaults to `12`):
            Number of bins per octave.

    Returns:
        `float` or `np.ndarray`: The frequencies on the octave scale.
    """
    stuttgart_pitch = 440.0 * 2.0 ** (tuning / bins_per_octave)
    octave = np.log2(freq / (float(stuttgart_pitch) / 16))
    return octave


def chroma_filter_bank(
    num_frequency_bins: int,
    num_chroma: int,
    sampling_rate: int,
    tuning: float = 0.0,
    power: float | None = 2.0,
    weighting_parameters: tuple[float, float] | None = (5.0, 2.0),
    start_at_c_chroma: bool = True,
):
    """
    Creates a chroma filter bank, i.e a linear transformation to project spectrogram bins onto chroma bins.

    Adapted from *librosa*.

    Args:
        num_frequency_bins (`int`):
            Number of frequencies used to compute the spectrogram (should be the same as in `stft`).
        num_chroma (`int`):
            Number of chroma bins (i.e pitch classes).
        sampling_rate (`float`):
            Sample rate of the audio waveform.
        tuning (`float`):
            Tuning deviation from A440 in fractions of a chroma bin.
        power (`float`, *optional*, defaults to 2.0):
            If 12.0, normalizes each column with their L2 norm. If 1.0, normalizes each column with their L1 norm.
        weighting_parameters (`tuple[float, float]`, *optional*, defaults to `(5., 2.)`):
            If specified, apply a Gaussian weighting parameterized by the first element of the tuple being the center and
            the second element being the Gaussian half-width.
        start_at_c_chroma (`bool`, *optional*, defaults to `True`):
            If True, the filter bank will start at the 'C' pitch class. Otherwise, it will start at 'A'.
    Returns:
        `np.ndarray` of shape `(num_frequency_bins, num_chroma)`
    """
    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sampling_rate, num_frequency_bins, endpoint=False)[1:]

    freq_bins = num_chroma * hertz_to_octave(frequencies, tuning=tuning, bins_per_octave=num_chroma)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    freq_bins = np.concatenate(([freq_bins[0] - 1.5 * num_chroma], freq_bins))

    bins_width = np.concatenate((np.maximum(freq_bins[1:] - freq_bins[:-1], 1.0), [1]))

    chroma_filters = np.subtract.outer(freq_bins, np.arange(0, num_chroma, dtype="d")).T

    num_chroma2 = np.round(float(num_chroma) / 2)

    # Project into range -num_chroma/2 .. num_chroma/2
    # add on fixed offset of 10*num_chroma to ensure all values passed to
    # rem are positive
    chroma_filters = np.remainder(chroma_filters + num_chroma2 + 10 * num_chroma, num_chroma) - num_chroma2

    # Gaussian bumps - 2*D to make them narrower
    chroma_filters = np.exp(-0.5 * (2 * chroma_filters / np.tile(bins_width, (num_chroma, 1))) ** 2)

    # normalize each column
    if power is not None:
        chroma_filters = chroma_filters / np.sum(chroma_filters**power, axis=0, keepdims=True) ** (1.0 / power)

    # Maybe apply scaling for fft bins
    if weighting_parameters is not None:
        center, half_width = weighting_parameters
        chroma_filters *= np.tile(
            np.exp(-0.5 * (((freq_bins / num_chroma - center) / half_width) ** 2)),
            (num_chroma, 1),
        )

    if start_at_c_chroma:
        chroma_filters = np.roll(chroma_filters, -3 * (num_chroma // 12), axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(chroma_filters[:, : int(1 + num_frequency_bins / 2)])


def optimal_fft_length(window_length: int) -> int:
    """
    Finds the best FFT input size for a given `window_length`. This function takes a given window length and, if not
    already a power of two, rounds it up to the next power or two.

    The FFT algorithm works fastest when the length of the input is a power of two, which may be larger than the size
    of the window or analysis frame. For example, if the window is 400 samples, using an FFT input size of 512 samples
    is more optimal than an FFT size of 400 samples. Using a larger FFT size does not affect the detected frequencies,
    it simply gives a higher frequency resolution (i.e. the frequency bins are smaller).
    """
    return 2 ** int(np.ceil(np.log2(window_length)))


def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: int | None = None,
    center: bool = True,
) -> np.ndarray:
    """
    Returns an array containing the specified window. This window is intended to be used with `stft`.

    The following window types are supported:

        - `"boxcar"`: a rectangular window
        - `"hamming"`: the Hamming window
        - `"hann"`: the Hann window
        - `"povey"`: the Povey window

    Args:
        window_length (`int`):
            The length of the window in samples.
        name (`str`, *optional*, defaults to `"hann"`):
            The name of the window function.
        periodic (`bool`, *optional*, defaults to `True`):
            Whether the window is periodic or symmetric.
        frame_length (`int`, *optional*):
            The length of the analysis frames in samples. Provide a value for `frame_length` if the window is smaller
            than the frame length, so that it will be zero-padded.
        center (`bool`, *optional*, defaults to `True`):
            Whether to center the window inside the FFT buffer. Only used when `frame_length` is provided.

    Returns:
        `np.ndarray` of shape `(window_length,)` or `(frame_length,)` containing the window.
    """
    length = window_length + 1 if periodic else window_length

    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    elif name == "povey":
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")

    if periodic:
        window = window[:-1]

    if frame_length is None:
        return window

    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )

    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window



# ═══════════════════════════════════════════════════════════════════════════════
# Audio math helpers (numpy/torch agnostic)
# ═══════════════════════════════════════════════════════════════════════════════


def _array_namespace(x):
    """Return the array module (``numpy`` or ``torch``) matching ``x``.

    Raises ``TypeError`` for unknown types. Use :func:`_xp_or_math` instead when
    Python scalars are also valid input.
    """
    if isinstance(x, np.ndarray):
        return np
    if is_torch_available():
        import torch

        if isinstance(x, torch.Tensor):
            return torch
    raise TypeError(f"Unsupported array type: {type(x)}")


def _xp_or_math(x):
    """Like :func:`_array_namespace` but returns ``math`` for Python scalars.

    Lets scalar-or-array math be written once: ``math.log10`` has the right
    signature for Python floats; numpy and torch use the same names on arrays.
    """
    if isinstance(x, (int, float)):
        return math
    return _array_namespace(x)


def _clamp_min(x, min_value):
    """Element-wise ``max(x, min_value)`` for numpy arrays or torch tensors.

    Needed because ``np.maximum(arr, scalar)`` accepts a Python scalar but
    ``torch.maximum(tensor, scalar)`` does not — and ``torch.clamp(x, min=)``
    has a different kwarg name than ``np.clip(x, a_min=)``.
    """
    if isinstance(x, np.ndarray):
        return np.maximum(x, min_value)
    return x.clamp(min=min_value)


def hertz_to_mel(freq: float | np.ndarray, mel_scale: str = "htk"):
    """
    Convert frequency from hertz to mels.

    Args:
        freq (`float`, `np.ndarray`, or `torch.Tensor`):
            The frequency, or multiple frequencies, in hertz (Hz).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        The frequencies on the mel scale, in the same form as the input.
    """
    if mel_scale not in ("htk", "kaldi", "slaney"):
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    xp = _xp_or_math(freq)

    if mel_scale == "htk":
        return 2595.0 * xp.log10(1.0 + freq / 700.0)
    if mel_scale == "kaldi":
        return 1127.0 * xp.log(1.0 + freq / 700.0)

    # slaney: linear below 1000 Hz, logarithmic above. The constants are written
    # differently per backend to preserve bit-exact parity with librosa (numpy) and
    # torchaudio (torch) — they use different float32 rounding paths.
    min_log_hertz = 1000.0
    min_log_mel = 15.0

    if xp is math:
        if freq >= min_log_hertz:
            return min_log_mel + math.log(freq / min_log_hertz) * 27.0 / math.log(6.4)
        return 3.0 * freq / 200.0

    if xp is np:
        linear = 3.0 * freq / 200.0
        logstep = 27.0 / np.log(6.4)
    else:  # torch — float32-tensor logstep matches torchaudio
        import torch

        linear = freq / (200.0 / 3.0)
        logstep = 27.0 / torch.log(torch.tensor(6.4))

    # Guard log against discarded-branch values; xp.where evaluates both branches.
    safe = _clamp_min(freq, min_log_hertz)
    log_branch = min_log_mel + xp.log(safe / min_log_hertz) * logstep
    return xp.where(freq >= min_log_hertz, log_branch, linear)


def mel_to_hertz(mels: float | np.ndarray, mel_scale: str = "htk"):
    """
    Convert frequency from mels to hertz.

    Args:
        mels (`float`, `np.ndarray`, or `torch.Tensor`):
            The frequency, or multiple frequencies, in mels.
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.

    Returns:
        The frequencies in hertz, in the same form as the input.
    """
    if mel_scale not in ("htk", "kaldi", "slaney"):
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

    xp = _xp_or_math(mels)

    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    if mel_scale == "kaldi":
        return 700.0 * (xp.exp(mels / 1127.0) - 1.0)

    # slaney — see note in hertz_to_mel; constants are written per-backend for
    # bit-exact parity with librosa (numpy) and torchaudio (torch).
    min_log_hertz = 1000.0
    min_log_mel = 15.0

    if xp is math:
        if mels >= min_log_mel:
            return min_log_hertz * math.exp(math.log(6.4) / 27.0 * (mels - min_log_mel))
        return 200.0 * mels / 3.0

    if xp is np:
        linear = 200.0 * mels / 3.0
        logstep = np.log(6.4) / 27.0
    else:  # torch — match old per-backend precision (Python-float logstep here,
        # though the reciprocal in hertz_to_mel uses a float32 tensor — old code
        # was inconsistent and we preserve that for bit-exact parity).
        linear = (200.0 / 3.0) * mels
        logstep = math.log(6.4) / 27.0

    log_branch = min_log_hertz * xp.exp(logstep * (mels - min_log_mel))
    return xp.where(mels >= min_log_mel, log_branch, linear)


def _create_triangular_filter_bank(fft_freqs, filter_freqs):
    """
    Triangular filter bank from FFT bin frequencies and filter center frequencies.

    Adapted from *torchaudio* and *librosa*. Works on numpy or torch inputs.

    Args:
        fft_freqs (array of shape `(num_frequency_bins,)`):
            Discrete frequencies of the FFT bins (in Hz, or in mel space when
            ``triangularize_in_mel_space=True``).
        filter_freqs (array of shape `(num_mel_filters + 2,)`):
            Edges and center frequencies of the triangular filters.

    Returns:
        Filter bank of shape `(num_frequency_bins, num_mel_filters)`.
    """
    xp = _array_namespace(fft_freqs)
    filter_diff = filter_freqs[1:] - filter_freqs[:-1]
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return _clamp_min(xp.minimum(down_slopes, up_slopes), 0)


def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: str | None = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """
    Creates a frequency bin conversion matrix used to obtain a mel spectrogram. This is called a *mel filter bank*, and
    various implementation exist, which differ in the number of filters, the shape of the filters, the way the filters
    are spaced, the bandwidth of the filters, and the manner in which the spectrum is warped. The goal of these
    features is to approximate the non-linear human perception of the variation in pitch with respect to the frequency.

    Different banks of mel filters were introduced in the literature. The following variations are supported:

    - MFCC FB-20: introduced in 1980 by Davis and Mermelstein, it assumes a sampling frequency of 10 kHz and a speech
      bandwidth of `[0, 4600]` Hz.
    - MFCC FB-24 HTK: from the Cambridge HMM Toolkit (HTK) (1995) uses a filter bank of 24 filters for a speech
      bandwidth of `[0, 8000]` Hz. This assumes sampling rate ≥ 16 kHz.
    - MFCC FB-40: from the Auditory Toolbox for MATLAB written by Slaney in 1998, assumes a sampling rate of 16 kHz and
      speech bandwidth of `[133, 6854]` Hz. This version also includes area normalization.
    - HFCC-E FB-29 (Human Factor Cepstral Coefficients) of Skowronski and Harris (2004), assumes a sampling rate of
      12.5 kHz and speech bandwidth of `[0, 6250]` Hz.

    This code is adapted from *torchaudio* and *librosa*. Note that the default parameters of torchaudio's
    `melscale_fbanks` implement the `"htk"` filters while librosa uses the `"slaney"` implementation.

    Args:
        num_frequency_bins (`int`):
            Number of frequency bins (should be the same as `n_fft // 2 + 1` where `n_fft` is the size of the Fourier Transform used to compute the spectrogram).
        num_mel_filters (`int`):
            Number of mel filters to generate.
        min_frequency (`float`):
            Lowest frequency of interest in Hz.
        max_frequency (`float`):
            Highest frequency of interest in Hz. This should not exceed `sampling_rate / 2`.
        sampling_rate (`int`):
            Sample rate of the audio waveform.
        norm (`str`, *optional*):
            If `"slaney"`, divide the triangular mel weights by the width of the mel band (area normalization).
        mel_scale (`str`, *optional*, defaults to `"htk"`):
            The mel frequency scale to use, `"htk"`, `"kaldi"` or `"slaney"`.
        triangularize_in_mel_space (`bool`, *optional*, defaults to `False`):
            If this option is enabled, the triangular filter is applied in mel space rather than frequency space. This
            should be set to `true` in order to get the same results as `torchaudio` when computing mel filters.

    Returns:
        `np.ndarray` of shape (`num_frequency_bins`, `num_mel_filters`): Triangular filter bank matrix. This is a
        projection matrix to go from a spectrogram to a mel spectrogram.
    """
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')

    if num_frequency_bins < 2:
        raise ValueError(f"Require num_frequency_bins: {num_frequency_bins} >= 2")

    if min_frequency > max_frequency:
        raise ValueError(f"Require min_frequency: {min_frequency} <= max_frequency: {max_frequency}")

    # center points of the triangular mel filters
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

    if triangularize_in_mel_space:
        # frequencies of FFT bins in Hz, but filters triangularized in mel space
        fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        # frequencies of FFT bins in Hz
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    if dtype is not None:
        # Per-band computation matching librosa's precision path: compute slopes in float64,
        # cast each band to dtype immediately. This replicates librosa's per-row assignment
        # to a dtype-initialized array, which produces different rounding than computing all
        # bands in float64 and casting at the end.
        filter_diff = np.diff(filter_freqs)
        ramps = np.subtract.outer(filter_freqs, fft_freqs)  # (num_mel_filters+2, num_frequency_bins)
        mel_filters = np.zeros((num_frequency_bins, num_mel_filters), dtype=dtype)
        for i in range(num_mel_filters):
            lower = -ramps[i] / filter_diff[i]
            upper = ramps[i + 2] / filter_diff[i + 1]
            mel_filters[:, i] = np.maximum(0, np.minimum(lower, upper)).astype(dtype)
    else:
        mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)

    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )

    return mel_filters


def power_to_db(
    spectrogram,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: float | None = None,
):
    """
    Converts a power spectrogram to the decibel scale. This computes `10 * log10(spectrogram / reference)`, using basic
    logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
    This means that large variations in energy may not sound all that different if the sound is loud to begin with.
    This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

    Based on the implementation of `librosa.power_to_db`. Works on numpy or torch inputs.

    Args:
        spectrogram (`np.ndarray` or `torch.Tensor`):
            The input power (mel) spectrogram. Note that a power spectrogram has the amplitudes squared!
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-10`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. The default of `1e-10` corresponds to a minimum of -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

    Returns:
        The spectrogram in decibels, same array type as the input.
    """
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = _clamp_min(spectrogram, min_value)
    spectrogram = 10.0 * (_array_namespace(spectrogram).log10(spectrogram) - math.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = _clamp_min(spectrogram, spectrogram.max() - db_range)

    return spectrogram


def amplitude_to_db(
    spectrogram,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: float | None = None,
):
    """
    Converts an amplitude spectrogram to the decibel scale. This computes `20 * log10(spectrogram / reference)`, using
    basic logarithm properties for numerical stability.

    The motivation behind applying the log function on the (mel) spectrogram is that humans do not hear loudness on a
    linear scale. Generally to double the perceived volume of a sound we need to put 8 times as much energy into it.
    This means that large variations in energy may not sound all that different if the sound is loud to begin with.
    This compression operation makes the (mel) spectrogram features match more closely what humans actually hear.

    Works on numpy or torch inputs.

    Args:
        spectrogram (`np.ndarray` or `torch.Tensor`):
            The input amplitude (mel) spectrogram.
        reference (`float`, *optional*, defaults to 1.0):
            Sets the input spectrogram value that corresponds to 0 dB. For example, use `np.max(spectrogram)` to set
            the loudest part to 0 dB. Must be greater than zero.
        min_value (`float`, *optional*, defaults to `1e-5`):
            The spectrogram will be clipped to this minimum value before conversion to decibels, to avoid taking
            `log(0)`. The default of `1e-5` corresponds to a minimum of -100 dB. Must be greater than zero.
        db_range (`float`, *optional*):
            Sets the maximum dynamic range in decibels. For example, if `db_range = 80`, the difference between the
            peak value and the smallest value will never be more than 80 dB. Must be greater than zero.

    Returns:
        The spectrogram in decibels, same array type as the input.
    """
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")

    reference = max(min_value, reference)

    spectrogram = _clamp_min(spectrogram, min_value)
    spectrogram = 20.0 * (_array_namespace(spectrogram).log10(spectrogram) - math.log10(reference))

    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = _clamp_min(spectrogram, spectrogram.max() - db_range)

    return spectrogram


