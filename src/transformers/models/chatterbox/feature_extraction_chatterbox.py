# coding=utf-8
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
"""Feature extractor class for Chatterbox."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided

from ...feature_extraction_sequence_utils import SequenceFeatureExtractor
from ...models.s3tokenizer.feature_extraction_s3tokenizer import S3TokenizerFeatureExtractor
from ...utils import is_librosa_available, is_torch_available
from ...utils.import_utils import requires


if is_torch_available():
    import torch
else:
    torch = None

if is_librosa_available():
    import librosa
else:
    librosa = None


class VoiceEncConfig:
    """
    Configuration for Voice Encoder.

    NOTE: This is intentionally aligned with `chatterbox.models.voice_encoder.config.VoiceEncConfig`.
    """

    num_mels = 40
    sample_rate = 16000
    speaker_embed_size = 256
    ve_hidden_size = 256
    flatten_lstm_params = False
    n_fft = 400
    hop_size = 160
    win_size = 400
    fmax = 8000
    fmin = 0
    preemphasis = 0.0
    mel_power = 2.0
    mel_type = "amp"
    normalized_mels = False
    ve_partial_frames = 160
    ve_final_relu = True
    stft_magnitude_min = 1e-4


@lru_cache
def _mel_basis_voice_encoder(hp: VoiceEncConfig):
    if librosa is None:
        raise ImportError(
            "librosa is required to compute mel filters for Chatterbox voice encoder preprocessing. "
            "Please install it with `pip install librosa`."
        )
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )  # (n_mels, 1 + n_fft//2)


def _preemphasis_voice_encoder(wav: np.ndarray, hp: VoiceEncConfig) -> np.ndarray:
    # Matches a simple lfilter([1, -preemph], [1], wav) without depending on scipy.
    assert hp.preemphasis != 0
    out = np.empty_like(wav)
    out[0] = wav[0]
    out[1:] = wav[1:] - hp.preemphasis * wav[:-1]
    return np.clip(out, -1, 1)


def _amp_to_db_voice_encoder(x: np.ndarray, hp: VoiceEncConfig) -> np.ndarray:
    return 20 * np.log10(np.maximum(hp.stft_magnitude_min, x))


def _normalize_voice_encoder(s: np.ndarray, hp: VoiceEncConfig, headroom_db: float = 15) -> np.ndarray:
    min_level_db = 20 * np.log10(hp.stft_magnitude_min)
    return (s - min_level_db) / (-min_level_db + headroom_db)


def _stft_voice_encoder(y: np.ndarray, hp: VoiceEncConfig, pad: bool = True):
    if librosa is None:
        raise ImportError(
            "librosa is required to compute STFT for Chatterbox voice encoder preprocessing. "
            "Please install it with `pip install librosa`."
        )
    # Match chatterbox: pad_mode="reflect" for historical/streaming consistency.
    return librosa.stft(
        y,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        center=pad,
        pad_mode="reflect",
    )


def melspectrogram_voice_encoder(wav: np.ndarray, hp: VoiceEncConfig, pad: bool = True) -> np.ndarray:
    """
    Voice encoder mel extraction aligned with `chatterbox.models.voice_encoder.melspec.melspectrogram`.

    Returns:
        np.ndarray of shape (num_mels, T)
    """
    if hp.preemphasis > 0:
        wav = _preemphasis_voice_encoder(wav, hp)
        assert np.abs(wav).max() - 1 < 1e-07

    spec_complex = _stft_voice_encoder(wav, hp, pad=pad)
    spec_magnitudes = np.abs(spec_complex)

    if hp.mel_power != 1.0:
        spec_magnitudes **= hp.mel_power

    mel = np.dot(_mel_basis_voice_encoder(hp), spec_magnitudes)
    if hp.mel_type == "db":
        mel = _amp_to_db_voice_encoder(mel, hp)

    if hp.normalized_mels:
        mel = _normalize_voice_encoder(mel, hp).astype(np.float32)

    assert not pad or mel.shape[1] == 1 + len(wav) // hp.hop_size
    return mel


def stride_as_partials(mel: np.ndarray, hp: VoiceEncConfig, overlap=0.5, rate: float | None = None, min_coverage=0.8):
    """Stride mel spectrogram into overlapping partials."""

    def get_frame_step(overlap, rate, hp):
        assert 0 <= overlap < 1
        if rate is None:
            frame_step = int(np.round(hp.ve_partial_frames * (1 - overlap)))
        else:
            frame_step = int(np.round((hp.sample_rate / rate) / hp.ve_partial_frames))
        assert 0 < frame_step <= hp.ve_partial_frames
        return frame_step

    def get_num_wins(n_frames, step, min_coverage, hp):
        assert n_frames > 0
        win_size = hp.ve_partial_frames
        n_wins, remainder = divmod(max(n_frames - win_size + step, 0), step)
        if n_wins == 0 or (remainder + (win_size - step)) / win_size >= min_coverage:
            n_wins += 1
        target_n = win_size + step * (n_wins - 1)
        return n_wins, target_n

    assert 0 < min_coverage <= 1
    frame_step = get_frame_step(overlap, rate, hp)
    n_partials, target_len = get_num_wins(len(mel), frame_step, min_coverage, hp)

    # Trim or pad
    if target_len > len(mel):
        mel = np.concatenate((mel, np.full((target_len - len(mel), hp.num_mels), 0)))
    elif target_len < len(mel):
        mel = mel[:target_len]

    mel = mel.astype(np.float32, order="C")
    shape = (n_partials, hp.ve_partial_frames, hp.num_mels)
    strides = (mel.strides[0] * frame_step, mel.strides[0], mel.strides[1])
    partials = as_strided(mel, shape, strides)
    return partials


@requires(backends=("torch",))
class ChatterboxFeatureExtractor(SequenceFeatureExtractor):
    """
    Constructs a Chatterbox feature extractor.

    This feature extractor is responsible for preparing *conditioning* inputs for `ChatterboxModel`, including:
    - resampling and truncation of reference audio,
    - `S3GenModel.embed_ref(...)` inputs,
    - voice-encoder speaker embedding inputs and extraction,
    - optional speech prompt tokenization via S3Tokenizer.

    Notes:
        This feature extractor is intentionally "model-assisted": it requires the instantiated sub-modules
        (`s3gen`, `voice_encoder`) to compute conditioning tensors aligned with the model weights.
    """

    # Not used for padding in the usual sense, but keep a minimal, consistent base configuration.
    model_input_names = [
        "speaker_emb",
        "cond_prompt_speech_tokens",
        "emotion_adv",
        "s3gen_ref_dict",
    ]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16000,
        padding_value: float = 0.0,
        s3gen_sampling_rate: int = 24000,
        s3gen_ref_seconds: int = 10,
        t3_prompt_seconds: int = 6,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.s3gen_sampling_rate = int(s3gen_sampling_rate)
        self.s3gen_ref_seconds = int(s3gen_ref_seconds)
        self.t3_prompt_seconds = int(t3_prompt_seconds)

        # Internal helper for prompt tokenization (16kHz mel features).
        self._s3_feature_extractor = S3TokenizerFeatureExtractor(sampling_rate=self.sampling_rate)

    @staticmethod
    def _to_1d_float32_np(reference_wav: Union[np.ndarray, list[float]]) -> np.ndarray:
        ref_np = np.asarray(reference_wav, dtype=np.float32)
        if ref_np.ndim == 0:
            raise ValueError("`reference_wav` must be a 1D waveform array, got a scalar.")
        if ref_np.ndim > 1:
            ref_np = ref_np.squeeze()
        if ref_np.ndim != 1:
            raise ValueError(f"`reference_wav` must be 1D after squeeze, got shape {ref_np.shape}.")
        return ref_np

    def extract_conditioning(
        self,
        reference_wav: Union[np.ndarray, list[float]],
        reference_sr: int,
        *,
        s3gen: Any,
        voice_encoder: Any,
        device: Union[str, torch.device],
        exaggeration: float = 0.5,
        speech_cond_prompt_len: int = 150,
    ) -> dict[str, Any]:
        """
        Extract conditioning for Chatterbox from a reference waveform.

        Args:
            reference_wav: Reference audio waveform (mono) as 1D numpy array or list of floats.
            reference_sr: Sampling rate (Hz) of `reference_wav`.
            s3gen: Instantiated S3Gen model. Must expose `embed_ref(...)` and `.tokenizer(...)`.
            voice_encoder: Instantiated voice encoder module. Must expose `embeds_from_wavs(...)`.
            device: Target device for returned torch tensors.
            exaggeration: Emotion/expressiveness level.
            speech_cond_prompt_len: Maximum length of speech conditioning prompt tokens. If <= 0, prompt is disabled.

        Returns:
            Dict with keys:
              - `speaker_emb`: torch.FloatTensor (1, speaker_embed_size)
              - `cond_prompt_speech_tokens`: Optional[torch.LongTensor] (1, prompt_len)
              - `emotion_adv`: torch.FloatTensor (1, 1, 1)
              - `s3gen_ref_dict`: dict produced by `s3gen.embed_ref(...)`
        """
        if reference_sr is None:
            raise ValueError("`reference_sr` must be provided for Chatterbox conditioning.")
        reference_sr = int(reference_sr)
        if reference_sr <= 0:
            raise ValueError(f"`reference_sr` must be > 0, got {reference_sr}.")

        ref_np = self._to_1d_float32_np(reference_wav)

        # Prepare audio for S3Gen (24kHz) and T3 components (16kHz).
        if librosa is None:
            raise ImportError(
                "librosa is required to resample reference audio for Chatterbox conditioning. "
                "Please install it with `pip install librosa`."
            )
        if reference_sr != self.s3gen_sampling_rate:
            ref_24k = librosa.resample(ref_np, orig_sr=reference_sr, target_sr=self.s3gen_sampling_rate)
        else:
            ref_24k = ref_np

        if reference_sr != self.sampling_rate:
            ref_16k = librosa.resample(ref_np, orig_sr=reference_sr, target_sr=self.sampling_rate)
        else:
            ref_16k = ref_np

        # Truncate for conditioning lengths.
        dec_len = self.s3gen_ref_seconds * self.s3gen_sampling_rate
        enc_len = self.t3_prompt_seconds * self.sampling_rate
        ref_24k = ref_24k[:dec_len]
        ref_16k_prompt = ref_16k[:enc_len]

        # Compute S3Gen conditioning dict.
        ref_tensor_24k = torch.from_numpy(ref_24k).unsqueeze(0).to(device)
        with torch.no_grad():
            s3gen_ref_dict = s3gen.embed_ref(ref_tensor_24k, self.s3gen_sampling_rate, device=device)

        # Voice encoder speaker embedding.
        ve_embed = voice_encoder.embeds_from_wavs([ref_16k], sample_rate=self.sampling_rate)
        speaker_emb = torch.from_numpy(ve_embed).to(device)

        # Speech prompt tokens for T3.
        cond_prompt_speech_tokens = None
        if int(speech_cond_prompt_len) > 0:
            features = self._s3_feature_extractor(
                ref_16k_prompt, sampling_rate=self.sampling_rate, return_tensors="pt"
            )
            features = features.to(device)
            with torch.no_grad():
                prompt_tokens, _ = s3gen.tokenizer(
                    input_features=features.input_features,
                    attention_mask=features.attention_mask,
                    return_dict=False,
                    max_len=int(speech_cond_prompt_len),
                )
            cond_prompt_speech_tokens = prompt_tokens.to(device)

        emotion_adv = float(exaggeration) * torch.ones(1, 1, 1, device=device)

        return {
            "speaker_emb": speaker_emb,
            "cond_prompt_speech_tokens": cond_prompt_speech_tokens,
            "emotion_adv": emotion_adv,
            "s3gen_ref_dict": s3gen_ref_dict,
        }


__all__ = [
    "ChatterboxFeatureExtractor",
]
