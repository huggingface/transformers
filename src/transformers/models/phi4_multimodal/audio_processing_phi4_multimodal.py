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

from ...audio_processing_backends import TorchAudioBackend
from ...audio_utils import mel_filter_bank
from ...feature_extraction_utils import BatchFeature


class Phi4MultimodalAudioProcessor(TorchAudioBackend):
    sample_rate = 16000
    force_mono = True
    preemphasis = 0.97
    n_fft = 512
    hop_length = 160
    win_length = 400
    n_mels = 80
    mel_min_frequency = 0
    mel_max_frequency = 7690
    audio_compression_rate = 8
    audio_downsample_rate = 1
    audio_feat_stride = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.n_mels,
            min_frequency=self.mel_min_frequency,
            max_frequency=self.mel_max_frequency,
            sampling_rate=self.sample_rate,
            triangularize_in_mel_space=True,
            mel_scale="kaldi",
        )

    def extract_spectrogram(self, audio, **kwargs):
        import torch

        waveform = torch.stack(audio)  # (batch, length)
        batch_size = waveform.shape[0]
        audio_lengths = kwargs.get("audio_lengths")

        fft_window = torch.hamming_window(self.win_length, periodic=False, dtype=torch.float64)
        frames = waveform.unfold(-1, self.win_length, self.hop_length)

        # Mask frames that overlap the boundary between real audio and padding
        if batch_size > 1 and audio_lengths is not None:
            frames = frames.clone()
            to_mask_batch_idxs = torch.arange(batch_size)[audio_lengths != audio_lengths.max()]
            if to_mask_batch_idxs.numel() > 0:
                batch_idxs_down = (audio_lengths[to_mask_batch_idxs] - self.win_length) // self.hop_length + 1
                batch_idxs_up = (audio_lengths[to_mask_batch_idxs] // self.hop_length) - 1
                offset_idx = batch_idxs_down.min()
                max_idx = batch_idxs_up.max()

                mask = torch.arange(max_idx - offset_idx).expand(to_mask_batch_idxs.shape[0], -1)
                mask = ((batch_idxs_down - offset_idx).unsqueeze(1) <= mask) & (
                    mask < (batch_idxs_up - offset_idx).unsqueeze(1)
                )
                mask = mask.unsqueeze(-1).expand(-1, -1, self.win_length)
                masked_frames = frames[to_mask_batch_idxs, offset_idx:max_idx].masked_fill_(mask, 0)
                frames[to_mask_batch_idxs, offset_idx:max_idx] = masked_frames

        # Pre-emphasis on frames with scaling
        frames_prev = torch.roll(frames, 1, dims=-1)
        frames_prev[:, :, 0] = frames_prev[:, :, 1]
        frames = (frames - self.preemphasis * frames_prev) * 32768

        # FFT
        S = torch.fft.rfft(fft_window * frames.view(-1, self.win_length), n=self.n_fft, dim=1)
        S = S.view(frames.shape[0], -1, S.shape[-1])
        S = S.to(torch.complex64)

        spec_power = torch.abs(S) ** 2

        # Mel filterbank + log
        mel_filters = torch.from_numpy(self.mel_filters).to(torch.float32)
        log_spec = torch.clamp(spec_power @ mel_filters, min=1.0)
        log_spec = torch.log(log_spec)

        return [log_spec[i] for i in range(batch_size)]

    def _compute_audio_embed_size(self, audio_frames):
        import torch

        integer = audio_frames // self.audio_compression_rate
        remainder = audio_frames % self.audio_compression_rate
        result = integer + (remainder > 0).to(integer.dtype)

        integer = result // self.audio_downsample_rate
        remainder = result % self.audio_downsample_rate
        result = integer + (remainder > 0).to(integer.dtype)

        return result

    def _preprocess(
        self,
        audio,
        padding,
        max_length,
        truncation,
        pad_to_multiple_of,
        return_tensors,
        **kwargs,
    ) -> BatchFeature:
        import torch

        # Capture original lengths before padding
        audio_lengths = torch.tensor([a.shape[-1] for a in audio])

        # Pad and truncate
        audio = self.pad(audio, padding, max_length, truncation, pad_to_multiple_of)

        # Extract spectrogram
        features = self.extract_spectrogram(audio, audio_lengths=audio_lengths)

        # Compute audio_embed_sizes
        feature_lengths = (audio_lengths - self.win_length) // self.hop_length + 1
        feature_lengths = feature_lengths * self.audio_feat_stride
        audio_embed_sizes = self._compute_audio_embed_size(feature_lengths)

        data = {"audio_features": features, "audio_embed_sizes": audio_embed_sizes}

        # Attention mask for batched inputs with different lengths
        if len(audio_lengths) > 1:
            feature_attention_mask = torch.arange(0, feature_lengths.max())[None, :] < feature_lengths[:, None]
            data["audio_attention_mask"] = feature_attention_mask

        output = BatchFeature(data, tensor_type=return_tensors)
        return output


__all__ = ["Phi4MultimodalAudioProcessor"]
