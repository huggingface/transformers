# Copyright 2026 The HuggingFace Inc. team.
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
from ...audio_utils import MelScaleConfig, SpectrogramConfig, StftConfig
from ...processing_utils import AudioKwargs


class Xcodec2AudioProcessorKwargs(AudioKwargs, total=False):
    r"""
    hop_length (`int`, *optional*, defaults to 320):
        Codec frame size, in samples. Distinct from the STFT hop in `spectrogram_config`.
    stride (`int`, *optional*, defaults to 2):
        Number of mel frames stacked into each output frame.
    feature_padding_value (`float`, *optional*, defaults to 1.0):
        Value used to pad the extracted features.
    """

    hop_length: int
    stride: int
    feature_padding_value: float


class Xcodec2AudioProcessorMixin:
    add_channel_dim = True
    do_extract_spectrogram = False
    # Mel frames are padded with 1.0 (the legacy FE's `padding_value`), unlike the raw audio
    force_mono = True
    # Legacy hub configs describe the fbank geometry with flat keys that are fixed
    # in the legacy config is the *mel* padding value; the raw audio is padded with 0.0.
    legacy_field_mapping = {
        "feature_size": None,
        "frame_length": None,
        "frame_shift": None,
        "num_mel_bins": None,
        "hop_length": None,
        "padding_value": "feature_padding_value",
    }
    model_input_names = ["audio_features", "audio_features_mask", "audio_values", "audio_values_mask"]
    pad_to_multiple_of = 320
    sampling_rate = 16000
    spectrogram_config = SpectrogramConfig(
        stft_config=StftConfig(
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn="povey",
            power=2.0,
            center=False,
            periodic=False,
            left_align_fft=True,
        ),
        mel_scale_config=MelScaleConfig(
            n_mels=80,
            f_min=20.0,
            f_max=8000.0,
            mel_scale="kaldi",
            triangularize_in_mel_space=True,
        ),
        log_mode="log",
        preemphasis=0.97,
        remove_dc_offset=True,
        mel_floor=1.192092955078125e-07,
        waveform_scale=32768.0,
    )

    hop_length = 320
    stride = 2
    feature_padding_value = 1.0
    valid_kwargs = Xcodec2AudioProcessorKwargs


class Xcodec2AudioProcessor(Xcodec2AudioProcessorMixin, TorchAudioBackend):
    def _process_audio(self, audio_el):
        # The legacy FE appends one zero sample to every waveform before padding
        audio_el = super()._process_audio(audio_el)
        return torch.nn.functional.pad(audio_el, (0, 1))

    def _postprocess_output(self, output, audio_ranges=None, **kwargs):
        audio_values = output["audio_values"]
        padded_length = audio_values.shape[-1]
        half_hop = self.hop_length // 2

        features = []
        for i, (start, end) in enumerate(audio_ranges):
            orig_length = end - start
            valid_length = min((orig_length + self.hop_length - 1) // self.hop_length * self.hop_length, padded_length)
            waveform = torch.nn.functional.pad(audio_values[i, 0, :valid_length], (half_hop, half_hop))
            f = self.extract_spectrogram([waveform], spectrogram_config=self.spectrogram_config)[0].transpose(-2, -1)
            f = (f - f.mean(0)) / torch.sqrt(f.var(0, unbiased=True) + 1e-7)
            features.append(f)

        frame_lengths = [f.shape[0] for f in features]
        max_frames = max(frame_lengths)
        if max_frames % self.stride:
            max_frames += self.stride - max_frames % self.stride
        batch = torch.stack(
            [
                torch.nn.functional.pad(f, (0, 0, 0, max_frames - f.shape[0]), value=self.feature_padding_value)
                for f in features
            ]
        )
        mask = self._get_mask([(0, length) for length in frame_lengths], max_frames)

        batch_size, num_frames, num_mel_bins = batch.shape
        output["audio_features"] = batch.reshape(batch_size, num_frames // self.stride, num_mel_bins * self.stride)
        output["audio_features_mask"] = (
            mask.reshape(batch_size, num_frames // self.stride, self.stride).min(dim=-1).values
        )
        return output


__all__ = ["Xcodec2AudioProcessor"]
