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

import math

import torch

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig


class GraniteSpeechAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    return_padding_mask = False
    do_extract_spectrogram = True
    projector_window_size = 15
    projector_downsample_rate = 5

    # Native pipeline, bit-equal to the upstream FE's `torchaudio.transforms.MelSpectrogram`
    # + log10 + Whisper-style max-clip/rescale (ADR 0004 post-log fields).
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            power=2.0,
        ),
        mel_scale_config=MelScaleConfig(n_mels=80),
        log_mode="log10",
        mel_floor=1e-10,
        clip_max_offset=8.0,
        post_log_shift=4.0,
        post_log_scale=0.25,
    )

    def extract_spectrogram(self, audio, **kwargs):
        logmel = super().extract_spectrogram(audio, **kwargs).transpose(-1, -2)  # (batch, time, n_mels)
        # Remove last frame if odd, then stack frame pairs
        if logmel.shape[1] % 2 == 1:
            logmel = logmel[:, :-1]
        return logmel.reshape(logmel.shape[0], -1, 2 * logmel.shape[-1])

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        hop_length = self.spectrogram_config.stft_config.hop_length

        # Compute audio_embed_sizes from original audio lengths
        effective_window_size = self.projector_window_size // self.projector_downsample_rate
        audio_embed_sizes = []
        for start, end in audio_ranges:
            raw_length = end - start
            mel_length = raw_length // hop_length + 1
            encoder_length = mel_length // 2
            nblocks = math.ceil(encoder_length / self.projector_window_size)
            projector_length = nblocks * effective_window_size
            audio_embed_sizes.append(projector_length)

        # Build input_features_mask matching the FE
        input_features_mask = torch.arange(max(audio_embed_sizes)).view(1, -1) < torch.tensor(
            audio_embed_sizes
        ).view(-1, 1)

        output["audio_embed_sizes"] = audio_embed_sizes
        output["audio_features_mask"] = input_features_mask
        return output


__all__ = ["GraniteSpeechAudioProcessor"]
