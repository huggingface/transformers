# Copyright 2026 IBM and The HuggingFace Team. All rights reserved.
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

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...processing_utils import AudioKwargs


class GraniteSpeech5AudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    delta_win_length (`int`, *optional*, defaults to 3):
        Window, in mel frames, of the delta (first-difference) filter stacked onto the log-mel
        features.
    frame_stacking (`int`, *optional*, defaults to 2):
        Number of consecutive mel frames concatenated into each encoder frame.
    """

    delta_win_length: int
    frame_stacking: int


class GraniteSpeech5AudioProcessorMixin:
    sampling_rate = 16000
    # `_finalize_output` builds its own mask over the frame-stacked length.
    return_padding_mask = False
    extra_model_input_names = ["audio_features_mask"]
    do_extract_spectrogram = True

    # Identical to granite_speech v4: the legacy extractor's `logmel_floor_db=8.0` floor followed
    # by `/4 + 1` is exactly this clip-and-rescale triple (ADR 0004 post-log fields).
    spectrogram_config = {
        "stft_config": {
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "power": 2.0,
        },
        "mel_scale_config": {"n_mels": 80},
        "log_mode": "log10",
        "mel_floor": 1e-10,
        "floor_below_peak": 8.0,
        "log_shift": 4.0,
        "log_scale": 0.25,
    }

    delta_win_length = 3
    frame_stacking = 2
    valid_kwargs = GraniteSpeech5AudioProcessorKwargs

    legacy_field_mapping = {
        # `feature_size` is *derived* here, not configured: the legacy extractor computes it as
        # `num_mel_bins * delta_expansion * frame_stacking` (80 * 2 * 2 = 320) and pops it from
        # kwargs. The base mapping sends `feature_size` to `n_mels` — correct for the extractors
        # that persisted the mel count under that name, wrong here — and it wins over the
        # `num_mel_bins` the same config carries, giving 320 mels and features 4x too wide.
        # Dropping it lets the real count through.
        "feature_size": None,
        "logmel_floor_db": "spectrogram_config.floor_below_peak",
        # Keys from the pre-`auto_map` remote-code configs that the *legacy extractor does not
        # read either*: its signature takes `num_mel_bins` and `sampling_rate`, and
        # `delta_expansion`/`frame_stacking` are class constants ("hardcoded in modeling for this
        # architecture"). Verified against it — a config saying `sample_rate=8000, n_mels=128,
        # stack_factor=8, deltas=False` still yields 16000/80/2. Honouring them would *break*
        # parity, so the drop is declared rather than left to look like an oversight.
        "n_mels": None,
        "sample_rate": None,
        "stack_factor": None,
        "deltas": None,
    }


class GraniteSpeech5AudioProcessor(GraniteSpeech5AudioProcessorMixin, TorchAudioBackend):
    def _compute_deltas(self, features, *, delta_win_length):
        """First-difference filter over time, matching ``torchaudio.functional.compute_deltas``.

        For a window of ``2n + 1`` the filter is ``sum(k * x[t+k]) / (n(n+1)(2n+1)/3)`` over
        ``k`` in ``[-n, n]``, with the signal replicate-padded at both ends. Implemented here
        rather than called from torchaudio, which the backends deliberately do not depend on.
        """
        n = (delta_win_length - 1) // 2
        denominator = n * (n + 1) * (2 * n + 1) / 3
        padded = torch.nn.functional.pad(features, (n, n), mode="replicate")
        kernel = torch.arange(-n, n + 1, dtype=features.dtype, device=features.device)
        kernel = kernel.expand(features.shape[-2], 1, -1)
        return torch.nn.functional.conv1d(padded, kernel, groups=features.shape[-2]) / denominator

    def _finalize_output(
        self, output, audio_ranges=None, *, frame_stacking, delta_win_length, spectrogram_config, **kwargs
    ):
        # (batch, n_mels, frames), already floored and rescaled by the spectrogram config.
        logmel = output.pop("audio_features")
        stacking = frame_stacking

        # The legacy extractor derives its frame count from the waveform as `len // hop` -- one
        # fewer than the centered STFT actually emits -- rounds that up to a whole stacking
        # group, and trims (or right-pads the waveform to reach it). Recover `len // hop` from
        # the emitted width rather than the waveform, which is no longer in scope here.
        num_frames = stacking * -(-(logmel.shape[-1] - 1) // stacking)
        if logmel.shape[-1] < num_frames:
            logmel = torch.nn.functional.pad(logmel, (0, num_frames - logmel.shape[-1]))
        else:
            logmel = logmel[..., :num_frames]

        logmel = torch.cat((logmel, self._compute_deltas(logmel, delta_win_length=delta_win_length)), dim=-2)
        logmel = logmel.transpose(-1, -2)
        batch_size = logmel.shape[0]
        output["audio_features"] = logmel.reshape(batch_size, -1, stacking * logmel.shape[-1])

        if audio_ranges is not None:
            hop = spectrogram_config.stft_config.hop_length
            lengths = torch.tensor(
                [-(-((end - start) // hop) // stacking) for start, end in audio_ranges],
                device=logmel.device,
            )
            max_frames = output["audio_features"].shape[1]
            mask = torch.arange(max_frames, device=logmel.device)[None, :] < lengths[:, None]
            output["audio_features_mask"] = mask.long()
        return output


__all__ = ["GraniteSpeech5AudioProcessor"]
